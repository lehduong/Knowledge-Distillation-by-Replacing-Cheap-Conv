from models.students.base_student import BaseStudent
import gc
import torch
import copy


class AuxStudent(BaseStudent):
    def __init__(self, model, distillation_args, aux_args, logger=None):
        """
        :param model: nn.Module object - pretrained model that need to be distilled
        :param distillation_args: LIST of DistillationArgs object - contains the distillation information
        """
        self.aux_layer_names = aux_args
        self.student_aux_outputs = []
        self.teacher_aux_outputs = []
        self._aux_hook_handlers = []
        super().__init__(model, distillation_args, logger)

    def _register_aux_hooks(self, aux_layer_names):
        """
            Register new list of auxiliary layer with the given names
                i.e. use when pruning new layer
        :param aux_layer_names: list of str
        :return:
        """
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

    def _remove_aux_hooks(self):
        """
        Remove the hooks of auxiliary layer
            i.e. used when changing to inference mode to save memory
        :return:
        """
        self.student_aux_outputs =[]
        self.teacher_aux_outputs = []
        while self._aux_hook_handlers:
            handler = self._aux_hook_handlers.pop()
            handler.remove()
        gc.collect()
        torch.cuda.empty_cache()

    def flush_aux_layers(self):
        """
        Completely remove all auxiliary layer
            i.e. used when pruning new layer
        :return:
        """
        self.aux_layer_names = []
        self._remove_aux_hooks()

    def update_aux_layers(self, aux_args):
        """
        Remove auxiliary layers and add new auxiliary layer
            i.e. user explicitly call this function when pruning new layer
        :param aux_args:
        :return:
        """
        self.flush_aux_layers()
        self._register_aux_hooks(aux_args)

    def forward(self, x):
        self.student_aux_outputs = []
        self.teacher_aux_outputs = []

        return super().forward(x)

    def inference(self, x):
        self.student_aux_outputs = []
        self.teacher_aux_outputs = []
        return super().inference(x)
    
    def eval(self):
        if not self.training:
            return self

        self.training = False

        for block in self.student_blocks:
            block.eval()
        self._remove_hooks()
        self._remove_aux_hooks()

        return self

    def train(self):
        """
        The parameters of teacher's network will always be set to EVAL
        :return: self
        """
        if self.training:
            return self

        self.training = True

        for block in self.student_blocks:
            block.train()

        if self._teaching:
            self._assign_blocks(student_mode=True)
        self._register_hooks()

        # create a copy as _register_aux_hooks will append the param to self.aux_layer_name
        aux_layer_names = copy.deepcopy(self.aux_layer_names)
        self.aux_layer_names = []
        self._register_aux_hooks(aux_layer_names)

        return self
