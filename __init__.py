from .stereovision import StereoscopicGenerator, AutostereogramGenerator

NODE_CLASS_MAPPINGS = {
    "StereoscopicGenerator": StereoscopicGenerator,
    "AutostereogramGenerator": AutostereogramGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StereoscopicGenerator": "🌀 Stereoscopic Generator",
    "AutostereogramGenerator": "🌀 Autostereogram Generator",
}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
