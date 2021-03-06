from typing import Any, Dict

from badger_utils.torch.device_aware import DeviceAwareModule


class SerializableModule(DeviceAwareModule):
    def serialize(self) -> Dict[str, Any]:
        return {
            'model': self.state_dict(),
        }

    def deserialize(self, data: Dict[str, Any]):
        self.load_state_dict(data['model'])
