from enum import Enum

from .cc3m_dataset import CC3MDataset
from .mmplanllm_dataset import MMPlanLLMDataset
from .vmr_dataset import VMRDataset


class DatasetType(str, Enum):
    MMPLANLLM_DATASET = "mmplanllm_dataset"
    CC3M_DATASET = "cc3m_dataset"
    VMR_DATASET = "vmr_dataset"

    def __str__(self):
        return self.value

    def __eq__(self, other):
        if isinstance(other, DatasetType):
            return self.value == other.value
        elif isinstance(other, str):
            return self.value == other
        return False


TYPE_TO_DATASET_CLASS = {
    DatasetType.MMPLANLLM_DATASET.value: MMPlanLLMDataset,
    DatasetType.CC3M_DATASET.value: CC3MDataset,
    DatasetType.VMR_DATASET.value: VMRDataset,
}
