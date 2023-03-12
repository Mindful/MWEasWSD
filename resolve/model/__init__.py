import sys, inspect
import resolve.model.components
component_classes = inspect.getmembers(sys.modules["resolve.model.components"], inspect.isclass)
COMPONENT_REGISTRY = {class_name: class_ref for (class_name, class_ref) in component_classes }
