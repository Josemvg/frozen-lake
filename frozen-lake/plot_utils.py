import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

def plot_environment(env, figsize=(5,4)):
    """
    Given a gym environment, plot the environment

    Parameters
    ----------
    env : gym environment
        The environment to plot
    figsize : tuple, optional
        The size of the figure to plot, by default (5,4)
    
    Returns
    -------
    None
        The plot is shown
    """
    plt.figure(figsize=figsize)
    img = env.render()
    plt.imshow(img)
    plt.axis("off")
    plt.show()
    return

def update_scene(num, frames, patch):
    """
    Update the scene for the animation

    Parameters
    ----------
    num : int
        The current frame number
    frames : list
        The list of frames to plot
    patch : matplotlib patch
        The patch to update

    Returns
    -------
    matplotlib patch
        The updated patch
    """
    patch.set_data(frames[num])
    return patch

def plot_animation(frames, repeat=False, interval=20):
    """
    Plot an animation of the frames

    Parameters
    ----------
    frames : list
        The list of frames to plot
    repeat : bool, optional
        Whether to repeat the animation, by default False
    interval : int, optional
        The interval between frames, by default 20
    
    Returns
    -------
    matplotlib animation
        The animation
    """
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    # Call the animator
    anim = animation.FuncAnimation(
        fig, update_scene, fargs=(frames, patch),
        frames=len(frames), repeat=repeat, interval=interval, cache_frame_data=False
    )
    plt.close()
    return anim

def save_gif(path, anim):
    """
    Save the animation as a gif

    Parameters
    ----------
    path : str
        The path to save the gif
    anim : matplotlib animation
        The animation to save

    Returns
    -------
    None
        The gif is saved
    """
    writergif = animation.PillowWriter(fps=30) 
    anim.save(path, writer=writergif)
    return