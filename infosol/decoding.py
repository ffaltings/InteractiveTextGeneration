import torch
from copy import deepcopy
from queue import PriorityQueue

class ParallelDecodingMixin():
    """
    Mixin for parallel decoding
    """

    def forward_canvases(self, canvases, device=torch.device('cpu'), move_to_cpu=True):
        raise NotImplemented

    @staticmethod
    def canvas_len(canvas):
        raise NotImplemented

    def decode_loop(self, canvases, decode_func, state=None, max_batch_tokens=2048, device=torch.device('cpu'), queue_size=2000, return_idx=False):

        iter_idx = 0
        finished = False
        input_queue = PriorityQueue(maxsize=queue_size)
        input_iter = iter(canvases)
        start_state = state

        def enqueue(canvas, state, iter_idx):
            input_queue.put((canvas, iter_idx, state))

        def add_input(idx):
            c = next(input_iter, None)
            if c is None:
                return True
            else:
                enqueue(deepcopy(c), start_state, idx)
                return False

        def get_batch():
            batch = []
            batch_len = 0
            batch_n_tokens = 0
            while True:
                if input_queue.empty(): break
                # get input
                inp = input_queue.get()
                canvas, idx, state = inp
                n_tokens = self.canvas_len(canvas)
                # check if can fit in batch
                batch_len_ = batch_len + 1
                batch_n_tokens_ = max(batch_n_tokens, n_tokens)
                # if doesn't fit, requeue and break
                if batch_len_ * batch_n_tokens_ > max_batch_tokens:
                    enqueue(canvas, state, idx) # todo: swith this order of arguments, it's confusing
                    break
                else:
                    batch.append(inp)
                    batch_len = batch_len_
                    batch_n_tokens = batch_n_tokens_
            batch_size = len(batch)
            if batch_size == 0:
                raise RuntimeError('Unable to fit any inputs into batch. Try increasing max batch tokens')
            return batch

        # start by filling up the queue
        while not input_queue.full():
            finished = add_input(iter_idx)
            iter_idx += 1
            if finished: break

        while not input_queue.empty():
            batch = get_batch()
            batch_size = len(batch)

            canvases = [b[0] for b in batch]
            model_out = self.forward_canvases(canvases, device=device, move_to_cpu=True)
            for i in range(batch_size):
                m_out, (canvas, idx, state) = model_out[i], batch[i]
                canvas, state, stop = decode_func(m_out, canvas, state)

                if stop:
                    if return_idx:
                        yield canvas, idx
                    else:
                        yield canvas
                    if not finished:
                        finished = add_input(iter_idx)
                        iter_idx += 1
                else:
                    enqueue(canvas, state, idx)
        return
