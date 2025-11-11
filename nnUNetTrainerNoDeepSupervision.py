from nnunetv2.training.nnUNetTrainer.nnUNetTrainer__ import nnUNetTrainer
import torch

from nnunetv2.training.nnUNetTrainer.variants.network_architecture.resnet.Attention_Unet import AttU_Net
from torch.nn.parallel import DistributedDataParallel as DDP
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels


class nnUNetTrainerNoDeepSupervision(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.enable_deep_supervision = False

    def initialize(self):
        if not self.was_initialized:
            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)

            # self.network = self.build_network_architecture(
            #     self.configuration_manager.network_arch_class_name,
            #     self.configuration_manager.network_arch_init_kwargs,
            #     self.configuration_manager.network_arch_init_kwargs_req_import,
            #     self.num_input_channels,
            #     self.label_manager.num_segmentation_heads,
            #     self.enable_deep_supervision
            # ).to(self.device)
            self.network = AttU_Net(in_channel=self.num_input_channels, num_classes=self.label_manager.num_segmentation_heads).to(self.device)
            print("=====================================================")
            print("NOW USE OUR Attention-UNet!")

            # compile network for free speedup
            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()
            # torch 2.2.2 crashes upon compiling CE loss
            # if self._do_i_compile():
            #     self.loss = torch.compile(self.loss)
            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")

    def set_deep_supervision_enabled(self, enabled: bool):
        pass