import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import seaborn as sns
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

def read_csv_new(filepath):

    data = pd.read_csv(filepath)
    psnr_avg = data['psnr_avg'].tolist()
    psnr_all = data['psnr_all'].tolist()
    psnr_variance = data['psnr_variance'].tolist()
    bpp_avg_est = data['bpp_avg_est'].tolist()
    bpp_avg = data['bpp_avg'].tolist()
    bpp_all = data['bpp_all'].tolist()
    bpp_variance = data['bpp_variance'].tolist()
    mse_avg = data['mse_avg'].tolist()
    mse_variance = data['mse_variance'].tolist()
    names = data['name'].tolist()
    
    # Extract band PSNR and MSE averages
    band_psnr_avg = {col: data[col].tolist() for col in data.columns if col.startswith('psnr_band_')}
    band_mse_avg = {col: data[col].tolist() for col in data.columns if col.startswith('mse_band_')}
    
    return names, psnr_avg, psnr_variance, bpp_avg_est, bpp_avg, bpp_variance, band_psnr_avg, mse_avg, mse_variance, band_mse_avg, psnr_all, bpp_all

def flatten(values):
    for item in values:
        if isinstance(item, list):
            for subitem in flatten(item):
                yield subitem
        else:
            yield item

def plot_rate_distortion(names, bpp, psnr, psnr_variance, bpp_variance, path, mode, fig_size=(12, 8)):

    plt.figure(dpi=600)
    _, axes = plt.subplots(1, 1, figsize=fig_size)

    # Convert all values to floats
    all_bpp_values = [float(value) for value in flatten(bpp.values())]
    all_psnr_values = [float(value) for value in flatten(psnr.values())]
    
    xlim = (min(all_bpp_values) * 0.85, max(all_bpp_values) * 1.1)
    ylim = (min(all_psnr_values) * 0.97, max(all_psnr_values) * 1.03)

    # Rate-Distortion Plot
    if mode == 'scatter':
        plt.figtext(.5, 0., '(upper-left is better)', fontsize=12, ha='center')
        for name in sorted(list(set(names))):
            # Plot faded scatter points
            axes.scatter(bpp[name], psnr[name], label=name) #, alpha=0.1)
            # print(len(bpp[name]))
            # print(len(bpp[name][0]))
            # bpp_values = np.array(bpp[name], dtype=float)
            # print()
            # psnr_values = np.array(psnr[name], dtype=float)

            # # Fit and plot a regression line (using polynomial regression for flexibility)
            # bpp_array = bpp_values.reshape(-1, 1)
            # psnr_array = psnr_values
            # model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
            # model.fit(bpp_array, psnr_array)
            # bpp_range = np.linspace(min(bpp[name]), max(bpp[name]), 100).reshape(-1, 1)
            # psnr_pred = model.predict(bpp_range)
            # axes.plot(bpp_range, psnr_pred, linestyle='-', linewidth=2, label=f'{name} (fit)')
    elif mode == 'error':
        for name in sorted(list(set(names))):
            axes.errorbar(bpp[name], psnr[name], xerr=bpp_variance[name], yerr=psnr_variance[name], linestyle='--', marker='o', label=name, linewidth=2, capsize=6)
        axes.set_title('Rate-Distortion with Standard Deviation Error Bars')
        error_info = 'Error bars represent ±1 standard deviation'
        plt.figtext(0.5, 0.02, error_info, wrap=True, horizontalalignment='center', fontsize=10)

    elif mode == 'normal':
        for name in sorted(list(set(names))):
            axes.plot(bpp[name], psnr[name], linestyle='--', marker='o', label=name, linewidth=2, alpha=0.85)

    axes.legend(loc='best')
    axes.set_ylabel('PSNR [dB]')
    axes.set_xlabel('Bit-rate [bpp]')
    axes.set_ylim(ylim)
    axes.set_xlim(xlim)
    axes.xaxis.grid(True, color="grey")
    axes.yaxis.grid(True, color="grey")

    plt.savefig(path)
    plt.close()

def plot_metrics(names, band_psnr_avg, band_mse_avg, overall_psnr_avg, overall_mse_avg, path, fig_size=(12, 14)):
    
    unique_names = list(set(names))

    # Determine common x and y limits
    all_psnr_values = []
    all_mse_values = []

    for name in unique_names:
        for band in band_psnr_avg:
            all_psnr_values.extend(band_psnr_avg[band][name])
        all_psnr_values.extend(overall_psnr_avg[name])

        for band in band_mse_avg:
            all_mse_values.extend(band_mse_avg[band][name])
        all_mse_values.extend(overall_mse_avg[name])

    psnr_ylim = (min(all_psnr_values) * 0.95, max(all_psnr_values)* 1.05)
    mse_ylim = (min(all_mse_values) * 0.95, max(all_mse_values)* 1.05)

    for name in unique_names:
    
        fig, axes = plt.subplots(2, 1, figsize=fig_size, dpi=300)

        # PSNR per Band subplot
        for band in band_psnr_avg:
            axes[0].plot(band_psnr_avg[band][name], linestyle='solid', marker='o', label=f'PSNR {band}')
        axes[0].plot(overall_psnr_avg[name], color='r', linestyle='--', label='Overall PSNR Mean')
        axes[0].set_ylabel('PSNR [dB]')
        axes[0].set_xlabel('Models')
        axes[0].grid(True, color="grey")
        axes[0].legend(loc='best')
        axes[0].set_title(f'PSNR per Band - {name}')
        axes[0].set_ylim(psnr_ylim)

        # MSE per Band subplot
        for band in band_mse_avg:
            axes[1].plot(band_mse_avg[band][name], linestyle='solid', marker='x', label=f'MSE {band}')
        axes[1].plot(overall_mse_avg[name], color='r', linestyle='--', label='Overall MSE Mean')
        axes[1].set_ylabel('MSE')
        axes[1].set_xlabel('Models')
        axes[1].grid(True, color="grey")
        axes[1].legend(loc='best')
        axes[1].set_title(f'MSE per Band - {name}')
        axes[1].set_ylim(mse_ylim)

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{path}_{name}.png")
        plt.close()


def evaluate_and_visualize_results(model_names, csv_path, output_dir):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

   
    names, psnr_avg, psnr_variance, bpp_avg_est, bpp_avg, bpp_variance, band_psnr_avg, mse_avg, mse_variance, band_mse_avg, psnr_all, bpp_all = read_csv_new(csv_path)

    # names.sort()

    # Filter data based on defined model_names
    indices = [i for i, name in enumerate(names) if name in model_names]
    
    names = [names[i] for i in indices]
    psnr_avg = [psnr_avg[i] for i in indices]
    psnr_variance = [psnr_variance[i] for i in indices]
    bpp_avg_est = [bpp_avg_est[i] for i in indices]
    bpp_avg = [bpp_avg[i] for i in indices]
    bpp_variance = [bpp_variance[i] for i in indices]
    mse_avg = [mse_avg[i] for i in indices]
    mse_variance = [mse_variance[i] for i in indices]
    psnr_all = [psnr_all[i] for i in indices]
    
    bpp_all = [bpp_all[i] for i in indices]
    band_psnr_avg = {k: [v[i] for i in indices] for k, v in band_psnr_avg.items()}
    band_mse_avg = {k: [v[i] for i in indices] for k, v in band_mse_avg.items()}

    psnr_dict = defaultdict(list)
    mse_dict = defaultdict(list)
    bpp_dict = defaultdict(list)
    psnr_all_dict= defaultdict(list)
    bpp_all_dict= defaultdict(list)
    psnr_var_dict = defaultdict(list)
    bpp_var_dict = defaultdict(list)
    band_psnr_avg_dict = defaultdict(list)
    band_mse_avg_dict = defaultdict(list)
    for k,_ in band_psnr_avg.items():   
        band_psnr_avg_dict[k] = defaultdict(list)
    for k,_ in band_mse_avg.items():
        band_mse_avg_dict[k] = defaultdict(list)


    # # Convert lists to dict for plotting
    for i in range(len(names)):
 
        if names[i] in psnr_dict:
            psnr_dict[names[i]].append(psnr_avg[i])
            mse_dict[names[i]].append(mse_avg[i])
            bpp_dict[names[i]].append(bpp_avg[i])
            psnr_all_dict[names[i]].append(psnr_all[i][1:-1].split(sep=','))
            bpp_all_dict [names[i]].append(bpp_all[i][1:-1].split(sep=','))
            psnr_var_dict[names[i]].append(np.sqrt(np.array(psnr_variance[i])))
            bpp_var_dict[names[i]].append(np.sqrt(np.array(bpp_variance[i])))
            for k,_ in band_psnr_avg.items():
                band_psnr_avg_dict[k][names[i]].append(band_psnr_avg[k][i])
            for k,_ in band_mse_avg.items():
                band_mse_avg_dict[k][names[i]].append(band_mse_avg[k][i])

        else: 
            psnr_dict[names[i]] = [psnr_avg[i]]
            mse_dict[names[i]] = [mse_avg[i]]
            bpp_dict[names[i]] = [bpp_avg[i]]
            psnr_all_dict[names[i]] = [psnr_all[i][1:-1].split(sep=',')]
            bpp_all_dict [names[i]] = [bpp_all[i][1:-1].split(sep=',')]
            psnr_var_dict[names[i]] = [np.sqrt(np.array(psnr_variance[i]))]
            bpp_var_dict[names[i]] = [np.sqrt(np.array(bpp_variance[i]))]

            for k,_ in band_psnr_avg.items():
                band_psnr_avg_dict[k][names[i]]= [band_psnr_avg[k][i]]

            for k,_ in band_mse_avg.items():

                band_mse_avg_dict[k][names[i]]= [band_mse_avg[k][i]]

    plot_rate_distortion(names, bpp_all_dict, psnr_all_dict,psnr_var_dict, bpp_var_dict,mode='scatter', path=f'{output_dir}/rate_distortion_scatter.png')
    plot_rate_distortion(names, bpp_dict, psnr_dict,psnr_var_dict, bpp_var_dict,mode='normal', path=f'{output_dir}/rate_distortion.png')
    plot_rate_distortion(names, bpp_dict, psnr_dict,psnr_var_dict, bpp_var_dict,mode='error', path=f'{output_dir}/rate_distortion_error.png')

    plot_metrics(names, band_psnr_avg_dict, band_mse_avg_dict, psnr_dict, mse_dict, path=f'{output_dir}/band_metrics')

def plot_correlation_matrix(file_path, path):

    df = pd.read_csv(file_path, index_col=0)
    
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # mask upper triangle.
    mask = np.triu(np.ones_like(df, dtype=bool))

    plt.figure(figsize=(8, 6), dpi=300)
    
    sns.heatmap(df,vmin=0,vmax=1, mask=mask, annot=True, fmt=".2f", cbar=True, 
                xticklabels=df.columns, yticklabels=df.index, square=True, 
                cbar_kws={"shrink": .75}, annot_kws={"fontsize": 8})
    
    plt.title('Band Correlation Matrix', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.savefig(path)
    plt.close()

def plot_per_image_correlations(file_path, path):

    df = pd.read_csv(file_path)
    
    df['Image_Index'] = df['Image_Index'].astype(int)
    df['Correlation'] = df['Correlation'].astype(float)
    
    # extract unique band combinations and sort them.
    band_combinations = df['Band_Combination'].unique()
    band_combinations_sorted = sorted(band_combinations, key=lambda x: (int(x.split('-')[0]), int(x.split('-')[1])))

    df['Band_Combination'] = pd.Categorical(df['Band_Combination'], categories=band_combinations_sorted, ordered=True)
    df = df.sort_values('Band_Combination')

    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(24, 8), dpi=300)  # increase width for better visibility.
    sns.boxplot(x='Band_Combination', y='Correlation', data=df, palette='viridis', legend=False)

    positions = [i for i, comb in enumerate(band_combinations_sorted) if int(comb.split('-')[0]) != int(band_combinations_sorted[i-1].split('-')[0])]
    for pos in positions:
        plt.axvline(x=pos - 0.5, color='gray', linestyle='--')

    plt.title('Distribution of Per-Image Correlations by Band Combination', fontsize=18, fontweight='bold')
    plt.xlabel('Band Combination', fontsize=14)
    plt.ylabel('Correlation', fontsize=14)
    plt.ylim(-1, 1) 
    plt.xticks(rotation=45, ha='right')
    plt.savefig(path)
    plt.close()