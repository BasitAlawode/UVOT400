import matplotlib.pyplot as plt
import numpy as np

from matplotlib import rc
rc('text', usetex=True)

import sys
sys.path.append("toolkit")

from basit_codes.draw_utils import COLOR, LINE_STYLE

def draw_success_precision(success_ret, name, videos, attr, precision_ret=None,
        norm_precision_ret=None, bold_name=None, axis=[0, 1], show_top=15, legend_cols=1):
    
    font_size = 10
    
    # success plot
    fig, ax = plt.subplots(figsize=(font_size,font_size))
    ax.grid(b=True)
    ax.set_aspect(1)
    plt.xlabel('Overlap threshold', fontsize=3*font_size)
    plt.ylabel('Success rate', fontsize=3*font_size)
    if attr == 'ALL':
        plt.title(r'\textbf{Success plots of OPE on %s}' % (name), fontsize=2*font_size)
    elif "test" in attr.lower():
        plt.title(r'\textbf{Success plots of OPE on %s}' % (attr), fontsize=2*font_size)
    else:
        plt.title(r'\textbf{Success plots of OPE - %s}' % (attr), fontsize=2*font_size)
    plt.axis([0, 1]+axis)

    tracker_color, tracker_linestyle = {}, {}

    success = {}
    thresholds = np.arange(0, 1.05, 0.05)
    for idx, tracker_name in enumerate(success_ret.keys()):
        value = [v for k, v in success_ret[tracker_name].items() if k in videos]
        success[tracker_name] = np.mean(value)
        #tracker_color[tracker_name] = COLOR[idx]
        #tracker_linestyle[tracker_name] = LINE_STYLE[idx]


    for idx, (tracker_name, auc) in  \
            enumerate(sorted(success.items(), key=lambda x:x[1], reverse=True)[:show_top]):
        if tracker_name == bold_name:
            label = r"\textbf{[%.3f] %s}" % (auc, tracker_name)
        else:
            label = "[%.3f] " % (auc) + tracker_name
        value = [v for k, v in success_ret[tracker_name].items() if k in videos]
        plt.plot(thresholds, np.mean(value, axis=0),
                color=COLOR[idx], linestyle=LINE_STYLE[idx],\
                        label=label, linewidth=2)

    ax.legend(loc='best', labelspacing=0.2, ncol=legend_cols, fontsize=1.6*font_size)
    ax.autoscale(enable=True, axis='both', tight=True)
    xmin, xmax, ymin, ymax = plt.axis()
    ax.autoscale(enable=False)
    ymax += 0.03
    ymin, ymax = 0, 1.01   # Added by me
    plt.grid(color = 'black', linestyle='dotted', linewidth=0.5)  # Added by me
    plt.axis([xmin, xmax, ymin, ymax])
    plt.xticks(np.arange(xmin, xmax+0.01, 0.1), fontsize=2*font_size)
    plt.yticks(np.arange(ymin, ymax, 0.1), fontsize=2*font_size)
    ax.set_aspect((xmax - xmin)/(ymax-ymin))
    #plt.show()
    
    plt.savefig(f"trackers_results/{name}/plots/success_plot_{name}.png", 
                    bbox_inches = 'tight', pad_inches = 0.05)
    plt.savefig(f"trackers_results/{name}/plots/success_plot_{name}.pdf", format="pdf", 
                    bbox_inches = 'tight', pad_inches = 0.05)
    plt.savefig(f"trackers_results/{name}/plots/success_plot_{name}.eps", format="eps", 
                    bbox_inches = 'tight', pad_inches = 0.05)

    if precision_ret:
        # norm precision plot
        fig, ax = plt.subplots(figsize=(font_size,font_size))
        ax.grid(b=True)
        ax.set_aspect(50)
        plt.xlabel('Location error threshold', fontsize=3*font_size)
        plt.ylabel('Precision', fontsize=3*font_size)
        if attr == 'ALL':
            plt.title(r'\textbf{Precision plots of OPE on %s}' % (name), fontsize=2*font_size)
        elif "test" in attr.lower():
            plt.title(r'\textbf{Precision plots of OPE on %s}' % (attr), fontsize=2*font_size)
        else:
            plt.title(r'\textbf{Precision plots of OPE - %s}' % (attr), fontsize=2*font_size)
        plt.axis([0, 50]+axis)
        precision = {}
        thresholds = np.arange(0, 51, 1)
        for tracker_name in precision_ret.keys():
            value = [v for k, v in precision_ret[tracker_name].items() if k in videos]
            precision[tracker_name] = np.mean(value, axis=0)[20]
        for idx, (tracker_name, pre) in \
                enumerate(sorted(precision.items(), key=lambda x:x[1], reverse=True)[:show_top]):
            if tracker_name == bold_name:
                label = r"\textbf{[%.3f] %s}" % (pre, tracker_name)
            else:
                label = "[%.3f] " % (pre) + tracker_name
            value = [v for k, v in precision_ret[tracker_name].items() if k in videos]
            plt.plot(thresholds, np.mean(value, axis=0),
                    color=COLOR[idx], linestyle=LINE_STYLE[idx],\
                        label=label, linewidth=2)
            
        ax.legend(loc='best', labelspacing=0.2, ncol=legend_cols, fontsize=1.6*font_size)
        ax.autoscale(enable=True, axis='both', tight=True)
        xmin, xmax, ymin, ymax = plt.axis()
        ax.autoscale(enable=False)
        ymax += 0.03
        ymin, ymax = 0, 1.01   # Added by me
        plt.grid(color = 'black', linestyle='dotted', linewidth=0.5)  # Added by me
        plt.axis([xmin, xmax, ymin, ymax])
        plt.xticks(np.arange(xmin, xmax+0.01, 5), fontsize=2*font_size)
        plt.yticks(np.arange(ymin, ymax, 0.1), fontsize=2*font_size)
        ax.set_aspect((xmax - xmin)/(ymax-ymin))
        #plt.show()
        plt.savefig(f"trackers_results/{name}/plots/precision_plot_{name}.png",
                      bbox_inches = 'tight', pad_inches = 0.05)
        plt.savefig(f"trackers_results/{name}/plots/precision_plot_{name}.pdf", format="pdf",
                      bbox_inches = 'tight', pad_inches = 0.05)
        plt.savefig(f"trackers_results/{name}/plots/precision_plot_{name}.eps", format="eps",
                      bbox_inches = 'tight', pad_inches = 0.05)

    # norm precision plot
    if norm_precision_ret:
        fig, ax = plt.subplots(figsize=(font_size,font_size))
        ax.grid(b=True)
        plt.xlabel('Location error threshold', fontsize=3*font_size)
        plt.ylabel('Normalized Precision', fontsize=3*font_size)
        if attr == 'ALL':
            plt.title(r'\textbf{Normalized Precision plots of OPE on %s}' % (name), 
                      fontsize=2*font_size)
        elif "test" in attr.lower():
            plt.title(r'\textbf{Normalized Precision plots of OPE on %s}' % (attr), 
                      fontsize=2*font_size)
        else:
            plt.title(r'\textbf{Normalized Precision plots of OPE - %s}' % (attr), 
                      fontsize=2*font_size)
        norm_precision = {}
        thresholds = np.arange(0, 51, 1) / 100
        for tracker_name in precision_ret.keys():
            value = [v for k, v in norm_precision_ret[tracker_name].items() if k in videos]
            norm_precision[tracker_name] = np.mean(value, axis=0)[20]
        for idx, (tracker_name, pre) in \
                enumerate(sorted(norm_precision.items(), key=lambda x:x[1], reverse=True)[:show_top]):
            if tracker_name == bold_name:
                label = r"\textbf{[%.3f] %s}" % (pre, tracker_name)
            else:
                label = "[%.3f] " % (pre) + tracker_name
            value = [v for k, v in norm_precision_ret[tracker_name].items() if k in videos]
            plt.plot(thresholds, np.mean(value, axis=0),
                    color=COLOR[idx], linestyle=LINE_STYLE[idx],\
                        label=label, linewidth=2)
        ax.legend(loc='best', labelspacing=0.2, ncol=legend_cols, fontsize=1.6*font_size)
        ax.autoscale(enable=True, axis='both', tight=True)
        xmin, xmax, ymin, ymax = plt.axis()
        ax.autoscale(enable=False)
        ymax += 0.03
        ymin, ymax = 0, 1.01   # Added by me
        plt.grid(color = 'black', linestyle='dotted', linewidth=0.5)  # Added by me
        plt.axis([xmin, xmax, ymin, ymax])
        plt.xticks(np.arange(xmin, xmax+0.01, 0.05), fontsize=2*font_size)
        plt.yticks(np.arange(ymin, ymax, 0.1), fontsize=2*font_size)
        ax.set_aspect((xmax - xmin)/(ymax-ymin))
        #plt.show()
        plt.savefig(f"trackers_results/{name}/plots/norm_precision_plot_{name}.png",
                      bbox_inches = 'tight', pad_inches = 0.05)
        plt.savefig(f"trackers_results/{name}/plots/norm_precision_plot_{name}.pdf", format="pdf",
                      bbox_inches = 'tight', pad_inches = 0.05)
        plt.savefig(f"trackers_results/{name}/plots/norm_precision_plot_{name}.eps", format="eps",
                      bbox_inches = 'tight', pad_inches = 0.05)
    
