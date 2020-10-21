import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C

GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 8.0


class ClipGradients(nn.Cell):
    """
    Clip gradients.

    Returns:
        List, a list of clipped_grad tuples.
    """

    def __init__(self):
        super(ClipGradients, self).__init__()
        self.clip_by_norm = nn.ClipByNorm()
        self.cast = P.Cast()
        self.dtype = P.DType()

    def construct(self,
                  grads,
                  clip_type,
                  clip_value):
        """
        Construct gradient clip network.

        Args:
            grads (list): List of gradient tuples.
            clip_type (Tensor): The way to clip, 'value' or 'norm'.
            clip_value (Tensor): Specifies how much to clip.

        Returns:
            List, a list of clipped_grad tuples.
        """
        if clip_type != 0 and clip_type != 1:  # pylint: disable=R1714
            return grads

        new_grads = ()
        for grad in grads:
            dt = self.dtype(grad)
            if clip_type == 0:
                t = C.clip_by_value(grad, self.cast(F.tuple_to_array((-clip_value,)), dt),
                                    self.cast(F.tuple_to_array((clip_value,)), dt))
            else:
                t = self.clip_by_norm(grad, self.cast(F.tuple_to_array((clip_value,)), dt))
            new_grads = new_grads + (t,)

        return new_grads