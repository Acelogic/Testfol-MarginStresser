import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import matplotlib.ticker as mtick

def apply_style():
    """Applies a professional aesthetic to matplotlib plots."""
    sns.set_theme(style="darkgrid", context="talk")
    plt.rcParams['figure.figsize'] = (14, 8)
    plt.rcParams['lines.linewidth'] = 2.0
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['font.family'] = 'sans-serif'
    
    # Custom Color Palette (Sim vs Real)
    # 0: Simulation (Vibrant Blue/Purple)
    # 1: Benchmark (Gray/Black)
    sns.set_palette(["#4C72B0", "#555555", "#C44E52", "#8172B2"])

def format_date_axis(ax):
    """Formats the X-axis for dates."""
    ax.xaxis.set_major_locator(mdates.YearLocator(2)) # Every 2 years
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=45)

def format_y_axis(ax, log=False):
    """Formats the Y-axis with commas and appropriate ticks."""
    if log:
        # Log Scale specific formatting
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
        # Add minor ticks for log scale visuals
        ax.yaxis.set_minor_formatter(mtick.NullFormatter())
    else:
        # Linear Scale
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))

def add_watermark(ax, text="NDXMEGA Simulation"):
    """Adds a subtle watermark."""
    ax.text(0.5, 0.5, text, transform=ax.transAxes,
            fontsize=40, color='gray', alpha=0.1,
            ha='center', va='center', rotation=30)
