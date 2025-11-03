from dataclasses import fields, is_dataclass


def print_dataclass_info_orig(obj, prefix=""):
    total_size = 0
    if not is_dataclass(obj):
        return
    for field in fields(obj):
        value = getattr(obj, field.name)
        field_path = f"{prefix}.{field.name}:" if prefix else field.name
        if is_dataclass(value):
            print_dataclass_info(value, prefix=field_path)  # recursive step
        else:
            info = f"type={type(value)}"
            if hasattr(value, "shape"):
                info += f"\n\tshape={value.shape}\n\tdtype={value.dtype}\n\tsize (MB)={round(value.nbytes * 1e-6, 3)}"
                total_size += (value.nbytes * 1e-6)
            else:
                info += f", value={value}"
            print(f"{field_path}: {info}")

    print(f"Total object size (MB): {total_size}")


def print_dataclass_info(obj, prefix=""):
    def _print(obj, prefix=""):
        total_size = 0
        if not is_dataclass(obj):
            return 0

        for field in fields(obj):
            value = getattr(obj, field.name)
            field_path = f"{prefix}.{field.name}" if prefix else field.name
            if is_dataclass(value):
                total_size += _print(value, prefix=field_path)
            else:
                info = f"type={type(value)}"
                if hasattr(value, "shape") and hasattr(value, "nbytes"):
                    size_mb = value.nbytes * 1e-6
                    info += f"\n\tshape={value.shape}\n\tdtype={value.dtype}\n\tsize (MB)={round(size_mb, 3)}"
                    total_size += size_mb
                else:
                    info += f", value={value}"
                print(f"{field_path}: {info}")
        return total_size

    total_size = _print(obj, prefix)
    print(f"Total object size (MB): {round(total_size, 3)}")
