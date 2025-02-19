import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from electrical_test_analyzer.helper_functions import ensure_iterable



default_categorical_cmap = 'tab10'
default_continuous_cmap = 'viridis'
default_marker_cycle = ['o', 's', '^', 'D', 'v', 'x', '*', 'p']
default_linestyle_cycle = ['-', '--', '-.', ':']


def create_mapping(unique_values, styles):
    return {val: styles[i % len(styles)] for i, val in enumerate(unique_values)}


def plot_df(
        df, x_key, y_key,
        x_func=lambda x : x, y_func=lambda y : y,
        group_by=None,
        color_by=None, cmap=None,
        linestyle_by=None, linestyle_cycle=None,
        marker_by=None, marker_cycle=None,
        label_by=None,
        split_yaxis_by=None, split_xaxis_by=None, ax=None,
        subplots_kw=None, scale_axes=True, plot_kw=None,
        split_yaxis_text_xy=(1.0, 0.5), split_yaxis_text_kwargs=None,
        split_xaxis_text_xy=(0.5, 1.0), split_xaxis_text_kwargs=None
    ):
    """Plots columns from a DataFrame.

    Args:
        df (pandas.Dataframe): The dataframe containing the data to plot.
        x_key (str): The name of the column containing the data to plot along the x axis.
        y_key (str): The name of the column containing the data to plot along the y axis.
        x_func (Callable[pandas.Series, Sequence, optional): A function to transform the x data. Defaults to identity function lambda x : x.
        y_func (Callable[pandas.Series, Sequence, optional): A function to transform the y data. Defaults to identity function lambda y : y.
        group_by (list[str], str, optional): The name of the column(s) by which the data is split. Defaults to None.
        color_by (str, optional): The name of the column by which colors are assigned to the plot. Defaults to None.
        cmap (matplotlib.colors.Colormap | list[tuple[float, float, float, float]], optional): A description of colors to be used. If None, a default value is used internally.
        linestyle_by (str, optional): The name of the column by which linestyles are assigned to the plot. Defaults to None.
        linestyle_cycle (list[str], optional): A description of the linestyles to be used. If None, a default value is used internally.
        marker_by (str, optional): The name of the column by which markers are assigned to the plot. Defaults to None.
        marker_cycle (list[str], optional): A descripton of the markers to be used. If None, a default value is used internally.
        label_by (str, optional): The name of the column by which labels are assigned. Defaults to None.
        split_yaxis_by (str, optional): The name of the columns by which the data is split into separate axes, stacked vertically. Defaults to None.
        split_xaxis_by (str, optional): The name of the columns by which the data is split into separate axes, stacked horizontally. Defaults to None.
        ax (matplotlib.axes.Axes, optional): The Axes on which to plot. Ignored if split_yaxis_by or split_xaxis_by are provided. Defaults to None.
        subplots_kw (dict, optional): Keyword arguments for matplotlib.pyplot.subplots call used to create the axes. Ignored if ax is used. Defaults to None.
        scale_axes (bool, optional): If true, the figure size is scaled by the numbers of rows and columns in the subplot. Defaults to True.
        plot_kw (dict, optional): Keyword arguments for plotting data, overwritten by column based plotting style. Defaults to None.
        split_yaxis_text_xy (tuple, optional): Coordinates for labeling split y axes in units of fractional axes. Defaults to (1.0, 0.5).
        split_yaxis_text_kwargs (dict, optional): Keyword arguments for text labeling split y axes. Defaults to None.
        split_xaxis_text_xy (tuple, optional): Coordinates for labeling split x axes in units of fractional axes. Defaults to (0.5, 1.0).
        split_xaxis_text_kwargs (dict, optional): Keyword arguments for text labeling split x axes. Defaults to None.

    Returns:
        tuple[matplotlib.figure, matplotlib.axes.Axes | numpy.ndarray[matplotlib.axes.Axes]]: A tuple containing:
            fig (matplotlib.figure): The figure.
            ax (matplotlib.axes.Axes | numpy.ndarray[matplotlib.axes.Axes]): The axes.
    """
    

    subplots_kw = subplots_kw or {}
    plot_kw = plot_kw or {}


    # Add all split by elements to group by

    group_by = group_by or []
    group_by = ensure_iterable(group_by)
    group_by.extend([x for x in [color_by, linestyle_by, marker_by, label_by, split_yaxis_by, split_xaxis_by] if x is not None and x not in group_by])

    if not group_by:
        if ax is not None:
            fig = ax.get_figure()
        else:
            fig, ax = plt.subplots(**subplots_kw)
        
        ax.plot(x_func(df[x_key]), y_func(df[y_key]), **plot_kw)

        return fig, ax

        
    # Set up mapping dictionaries

    if color_by is not None:
        color_ind = group_by.index(color_by)
        unique_color_values = df[color_by].unique()

        # Set up colormap

        if isinstance(cmap, mcolors.Colormap):
            pass
        elif isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)
        elif isinstance(cmap, list):
            cmap = mcolors.ListedColormap(cmap)
        else:
            if np.issubdtype(df[color_by].dtype, np.number):
                cmap = plt.get_cmap(default_continuous_cmap)
            else:
                cmap = plt.get_cmap(default_categorical_cmap)



        # if cmap is None:
        #     if np.issubdtype(df[color_by].dtype, np.number):
        #         norm = mcolors.Normalize(vmin=unique_color_values.min(), vmax=unique_color_values.max())
        #         cmap_obj = plt.get_cmap(default_continuous_cmap)
        #         color_map = create_mapping(unique_color_values, [cmap_obj(norm(color_value)) for color_value in unique_color_values])
        #     else:
        #         color_list = plt.get_cmap(default_categorical_cmap).colors
        #         color_map = create_mapping(unique_color_values, color_list)
        


        if isinstance(cmap, mcolors.ListedColormap):
            color_list = cmap.colors
            color_map = create_mapping(unique_color_values, color_list)
        else:
            if np.issubdtype(df[color_by].dtype, np.number):
                norm = mcolors.Normalize(vmin=unique_color_values.min(), vmax=unique_color_values.max())
                color_map = create_mapping(unique_color_values, [cmap(norm(color_value)) for color_value in unique_color_values])
            else:
                print(f'{unique_color_values=}')
                print(f'{len(unique_color_values)=}')
                print(f'{np.linspace(0, 1, len(unique_color_values))=}')
                color_list = cmap(np.linspace(0, 1, len(unique_color_values)))
                print(f'{color_list=}')
                color_map = create_mapping(unique_color_values, color_list)
        

    if linestyle_by is not None:
        linestyle_ind = group_by.index(linestyle_by)
        linestyle_cycle = linestyle_cycle or default_linestyle_cycle
        linestyle_map = create_mapping(df[linestyle_by].unique(), linestyle_cycle)
    
    if marker_by is not None:
        marker_ind = group_by.index(marker_by)
        marker_cycle = marker_cycle or default_marker_cycle
        marker_map = create_mapping(df[marker_by].unique(), marker_cycle)

    if label_by is not None:
        label_ind = group_by.index(label_by)


    if split_yaxis_by is not None:
        split_yaxis_ind = group_by.index(split_yaxis_by)
        yaxes_map = {val : i for i, val in enumerate(df[split_yaxis_by].unique())}
        nrows = len(yaxes_map)
    else:
        nrows = 1
    

    if split_xaxis_by is not None:
        split_xaxis_ind = group_by.index(split_xaxis_by)
        xaxes_map = {val : i for i, val in enumerate(df[split_xaxis_by].unique())}
        ncols = len(xaxes_map)
    else:
        ncols = 1


    # Create / get figure

    figsize = subplots_kw.get('figsize', (6.4, 4.8))
    if scale_axes:
        figsize = tuple([scale * length for scale, length in zip((ncols, nrows), figsize)])
    
    subplots_kw = {**subplots_kw, **dict(nrows=nrows, ncols=ncols, figsize=figsize)}

    if split_yaxis_by is not None and split_xaxis_by is not None:
        fig, ax = plt.subplots(**subplots_kw)
        ax = np.atleast_2d(ax)
    elif split_yaxis_by is not None:
        fig, ax = plt.subplots(**subplots_kw)
        ax = np.atleast_1d(ax)
    elif split_xaxis_by is not None:
        fig, ax = plt.subplots(**subplots_kw)
        ax = np.atleast_1d(ax)
    else:
        if ax is not None:
            fig = ax.get_figure()
        else:
            fig, ax = plt.subplots(**subplots_kw)
    

    # Plot

    for tags, subset_df in df.groupby(group_by):
        
        # Get appropriate ax

        if split_yaxis_by is not None and split_xaxis_by is not None:
            current_ax = ax[yaxes_map[tags[split_yaxis_ind]], xaxes_map[tags[split_xaxis_ind]]]
        elif split_yaxis_by is not None:
            current_ax = ax[yaxes_map[tags[split_yaxis_ind]]]
        elif split_xaxis_by is not None:
            current_ax = ax[xaxes_map[tags[split_xaxis_ind]]]
        else:
            current_ax = ax


        # Set up plot properties

        additional_plot_kw = {}
        if color_by is not None:
            additional_plot_kw['color'] = color_map[tags[color_ind]]
        if linestyle_by is not None:
            additional_plot_kw['linestyle'] = linestyle_map[tags[linestyle_ind]]
        if marker_by is not None:
            additional_plot_kw['marker'] = marker_map[tags[marker_ind]]
        if label_by is not None:
            additional_plot_kw['label'] = tags[label_ind]

        current_ax.plot(x_func(subset_df[x_key]), y_func(subset_df[y_key]), **{**plot_kw, **additional_plot_kw})
    

    # Add split plot labels

    if split_yaxis_by is not None:
        if split_yaxis_text_xy is not None:
            for val, i in yaxes_map.items():
                index = (i,) + (ax.ndim - 1) * (-1,)
                split_yaxis_text_kwargs_instance = split_yaxis_text_kwargs or {}
                split_yaxis_text_kwargs_instance = {**dict(transform=ax[index].transAxes, rotation=-90, va='center', ha='left'), **split_yaxis_text_kwargs_instance}
                ax[index].text(*split_yaxis_text_xy, val, **split_yaxis_text_kwargs_instance)
    
    if split_xaxis_by is not None:
        if split_xaxis_text_xy is not None:
            for val, i in xaxes_map.items():
                index = (ax.ndim - 1) * (0,) + (i,)
                split_xaxis_text_kwargs_instance = split_xaxis_text_kwargs or {}
                split_xaxis_text_kwargs_instance = {**dict(transform=ax[index].transAxes, va='bottom', ha='center'), **split_xaxis_text_kwargs_instance}
                ax[index].text(*split_xaxis_text_xy, val, **split_xaxis_text_kwargs_instance)


    return fig, ax