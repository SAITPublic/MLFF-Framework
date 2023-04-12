
import importlib

from ocpmodels.common.registry import _get_absolute_mapping


class MDEvaluateRegistry:
    r"""Class for registry object which acts as central source of truth."""
    mapping = {
        # Mappings to respective classes.
        "md_evaluate_name_mapping": {},
        "state": {},
    }

    @classmethod
    def register_md_evaluate(cls, name):
        r"""Register a trainer to registry with key 'name'

        Args:
            name: Key with which the trainer will be registered.

        Usage::

            from src.common.registry import registry

            @registry.register_md_evaluate("active_evaluator")
            class ActiveEvaluator():
                ...
        """

        def wrap(func):
            cls.mapping["md_evaluate_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register(cls, name, obj):
        r"""Register an item to registry with key 'name'

        Args:
            name: Key with which the item will be registered.

        Usage::

            from src.common.registry import registry

            registry.register("config", {})
        """
        path = name.split(".")
        current = cls.mapping["state"]

        for part in path[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[path[-1]] = obj

    @classmethod
    def __import_error(cls, name: str, mapping_name: str):
        kind = mapping_name[: -len("_name_mapping")]
        mapping = cls.mapping.get(mapping_name, {})
        existing_keys = list(mapping.keys())

        existing_cls_path = (
            mapping.get(existing_keys[-1], None) if existing_keys else None
        )
        if existing_cls_path is not None:
            existing_cls_path = f"{existing_cls_path.__module__}.{existing_cls_path.__qualname__}"
        else:
            existing_cls_path = "src.md_evaluate.base_evaluator.BenchmarkEvaluator"
        
        existing_keys = [f"'{name}'" for name in existing_keys]
        existing_keys = (
            ", ".join(existing_keys[:-1]) + " or " + existing_keys[-1]
        )
        existing_keys_str = (
            f" (one of {existing_keys})" if existing_keys else ""
        )
        return RuntimeError(
            f"Failed to find the {kind} '{name}'. "
            f"You may either use a {kind} from the registry{existing_keys_str} "
            f"or provide the full import path to the {kind} (e.g., '{existing_cls_path}')."
        )

    @classmethod
    def get_class(cls, name: str, mapping_name: str):
        existing_mapping = cls.mapping[mapping_name].get(name, None)
        if existing_mapping is not None:
            return existing_mapping

        # mapping be class path of type `{module_name}.{class_name}` (e.g., `src.trainers.forces_trainer.ForcesTrainer`)
        if name.count(".") < 1:
            raise cls.__import_error(name, mapping_name)

        try:
            return _get_absolute_mapping(name)
        except RuntimeError as e:
            raise cls.__import_error(name, mapping_name) from e

    @classmethod
    def get_md_evaluate_class(cls, name):
        return cls.get_class(name, "md_evaluate_name_mapping")

    @classmethod
    def get(cls, name, default=None, no_warning=False):
        r"""Get an item from registry with key 'name'

        Args:
            name (string): Key whose value needs to be retreived.
            default: If passed and key is not in registry, default value will
                     be returned with a warning. Default: None
            no_warning (bool): If passed as True, warning when key doesn't exist
                               will not be generated. Useful for cgcnn's
                               internal operations. Default: False
        Usage::

            from src.common.registry import registry

            config = registry.get("config")
        """
        original_name = name
        name = name.split(".")
        value = cls.mapping["state"]
        for subname in name:
            value = value.get(subname, default)
            if value is default:
                break

        if (
            "writer" in cls.mapping["state"]
            and value == default
            and no_warning is False
        ):
            cls.mapping["state"]["writer"].write(
                "Key {} is not present in registry, returning default value "
                "of {}".format(original_name, default)
            )
        return value

    @classmethod
    def unregister(cls, name):
        r"""Remove an item from registry with key 'name'

        Args:
            name: Key which needs to be removed.
        Usage::

            from src.common.registry import registry

            config = registry.unregister("config")
        """
        return cls.mapping["state"].pop(name, None)


md_evaluate_registry = MDEvaluateRegistry()
