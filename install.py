import launch

if not launch.is_installed("lavis"):
    launch.run_pip(
        "install salesforce-lavis",
        "requirements for BLIP2 captioner (lavis)",
    )

if not launch.is_installed("pywintypes"):
    launch.run_pip(
        "install pywintypes",
        "requirements for BLIP2 captioner (pywintypes)",
    )