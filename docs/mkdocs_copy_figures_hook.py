import shutil


def copy_figures(*args, **kwargs):
    """
    Copy the figures from the reports folder to the site folder
    :param site:
    :param config:
    :return:
    """
    shutil.copytree("reports/figures", "docs/reports/figures", dirs_exist_ok=True)
