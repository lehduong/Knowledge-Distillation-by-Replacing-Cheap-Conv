from models.students.base_student import BaseStudent
import gc
import torch


class AuxStudent(BaseStudent):
    def __init__(self, model, distillation_args, aux_args, logger=None):
        """
        :param model: nn.Module object - pretrained model that need to be distilled
        :param distillation_args: LIST of DistillationArgs object - contains the distillation information
        """
        super().__init__(model, distillation_args, logger)
        self.aux_layer_names = aux_args
        self.student_aux_outputs = []
        self.teacher_aux_outputs = []
        self._aux_hook_handlers = []

    def _register_aux_hooks(self, aux_layer_names):
        def save_forward_output(out):
            if self._teaching:
                self.teacher_aux_outputs.append(out)
            else:
                self.student_aux_outputs.append(out)

        # register hook for saving hidden outputs
        for layer_name in aux_layer_names:
            block = self.get_block(layer_name)
            handler = block.register_forward_hook(lambda m, inp, out: save_forward_output(out))
            self._aux_hook_handlers.append(handler)

        self.aux_layer_names = self.aux_layer_names + aux_layer_names

    def flush_aux_layers(self):
        self.aux_layer_names = []
        self.student_aux_outputs = []
        self.teacher_aux_outputs = []
        while self._aux_hook_handlers:
            handler = self._aux_hook_handlers.pop()
            handler.remove()

        gc.collect()
        torch.cuda.empty_cache()

    def update_aux_layers(self, aux_args):
        self.flush_aux_layers()
        self.aux_layer_names = aux_args
        self._register_aux_hooks(aux_args)
