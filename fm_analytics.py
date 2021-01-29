import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('./fm_analysis_1000.csv').sort_values(by='true_count')

sns.set(font_scale=1.5,
       rc={'axes.facecolor': 'lightgrey'})

g = sns.FacetGrid(df, sharex=False, sharey=False, col='wiki_term',
                 height=8, aspect=1, col_wrap=2, margin_titles=True)
g.map(sns.histplot, 'pcsa_count', color='g').set(yscale='log');

def vertical_mean_line(x, **kwargs):
    
    if x.name == 'true_count':
        name = "True Value"
        c = 'r'
        plt.axvline(x.mean(), linestyle="--", color=c)
    elif x.name == 'fm_count':
        name = 'Flajolet Count'
        c = 'b'
        plt.axvline(x.mean(), linestyle="--", color=c)
        txkw = dict(size=16, color=c, fontfamily='monospace')
        tx = name+": {:.0f}".format(x.mean())
        plt.text(x.mean()+25, 25, tx, **txkw)
    else:
        c = 'g'
        name= 'PCSA Mean'
        plt.axvline(x.mean(), linestyle="--", color=c)
        txkw = dict(size=16, color=c, fontfamily='monospace')
        tx = name+": {:.0f}".format(x.mean())
        plt.text(x.mean()+25, 100, tx, **txkw)

    
def text_box(x, **kwargs):
    txkw = dict(size=12, color = 'r', fontfamily='monospace')
    mean = x.mean()
    tx = "True Number: {:.0f}".format(mean)
    ax = plt.gca()
    ax.text(0.7, 0.9, tx, **txkw,
            transform=ax.transAxes);
        
    
g.map(vertical_mean_line, 'true_count')
g.map(vertical_mean_line, 'fm_count')
g.map(vertical_mean_line, 'pcsa_count')
g.map(text_box, 'true_count')
g.fig.subplots_adjust(top=0.92)
g.fig.suptitle('Length Estimation Wikipedia Entries') 
g.set_axis_labels( "Distribution of Counts" , "Counts")
g.savefig('distribution.png')
