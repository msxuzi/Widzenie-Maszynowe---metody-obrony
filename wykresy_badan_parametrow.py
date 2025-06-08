import pandas as pd
import matplotlib.pyplot as plt
import os

# Ścieżka do folderu z plikami CSV
input_dir = r"C:\PYTHON\WM_proj\eksport_wyników"

# Lista plików do analizy
filenames = [
    "grid_search_base_model_fgsm.csv",
    "grid_search_adv_model_fgsm.csv",
    "grid_search_base_model_pgd.csv",
    "grid_search_adv_model_pgd.csv"
]

# Style markerów (opcjonalne)
marker_styles = ['o', 's', 'x', '^', '*', 'D', 'P', 'H']

for file in filenames:
    path = os.path.join(input_dir, file)
    if not os.path.exists(path):
        print(f"Pominięto: {file} (brak pliku)")
        continue

    df = pd.read_csv(path)
    model_name = df['model'].iloc[0]
    attack_type = file.split('_')[-1].split('.')[0]

    for defense in ['blur', 'bit', 'median', 'jpeg']:
        df_def = df[df['defense'] == defense]
        if df_def.empty:
            continue

        plt.figure(figsize=(8, 6))

        for i, (param_val, group) in enumerate(df_def.groupby('param_value')):
            group_sorted = group.sort_values('epsilon')
            label = f"{defense} ({group['param_name'].iloc[0]}={param_val})"
            marker = marker_styles[i % len(marker_styles)]
            plt.plot(group_sorted['epsilon'], group_sorted['accuracy'], marker=marker, label=label)

        plt.title(f"{model_name.upper()} – {attack_type.upper()} – Obrona: {defense}")
        plt.xlabel("Epsilon")
        plt.ylabel("Accuracy (%)")
        plt.ylim(0, 100)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # Zapis wykresu
        plot_name = f"{file.replace('.csv', '')}_{defense}.png"
        plot_path = os.path.join(input_dir, plot_name)
        plt.savefig(plot_path)
        print(f"Zapisano wykres: {plot_path}")
        plt.close()
