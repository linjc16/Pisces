# fmt: off

from ase.visualize.plot import animate


def write_animation(filename, images, writer=None,
                    interval=200,
                    save_count=None,  # ignored for newer matplotlib (2023)
                    save_parameters=None, ax=None, **kwargs):
    import matplotlib.pyplot as plt

    if save_parameters is None:
        save_parameters = {}

    if ax is None:
        ax = plt.gca()

    animation = animate(images, ax=ax,
                        interval=interval,
                        **kwargs)
    animation.save(filename, writer=writer,
                   **save_parameters)


# Shortcuts for ase.io.formats (guessing file type from extension):
write_gif = write_animation
write_mp4 = write_animation
