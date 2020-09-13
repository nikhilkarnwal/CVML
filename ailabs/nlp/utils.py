import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def getattr(obj):
    return {name: obj.__getattribute__(name) for name in dir(obj) if not name.startswith('_')}


def print_obj(obj):
    print(json.dumps(getattr(obj), indent=4))


plt.switch_backend('agg')


def show_plot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
