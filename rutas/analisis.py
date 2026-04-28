import base64
from io import BytesIO

@app.route("/analisis", methods=["GET", "POST"])
def analisis():
    if not login_required():
        return redirect(url_for("login"))

    if request.method == "GET":
        return render_template("analisis.html", resultado=None,
                               filename=None, opciones={})

    archivo           = request.files.get("archivo")
    eliminar_outliers = request.form.get("eliminar_outliers") == "1"
    opciones          = {"eliminar_outliers": eliminar_outliers}

    if not archivo or archivo.filename == "":
        flash("No se seleccionó archivo")
        return render_template("analisis.html", resultado=None,
                               filename=None, opciones=opciones)

    try:
        import pandas as pd
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        from scipy import stats

        df = pd.read_csv(archivo)

        if "label" not in df.columns:
            flash("El CSV debe tener una columna 'label'")
            return render_template("analisis.html", resultado=None,
                                   filename=None, opciones=opciones)

        COLORES    = ['#5b8fb9','#ef4444','#16a34a','#f59e0b',
                      '#8b5cf6','#ec4899','#14b8a6','#f97316']
        feat_cols  = [c for c in ["WL","RMS","MAV","WAMP"] if c in df.columns]
        outliers_info = None

        # ── Eliminar outliers (Z-score < 3) ──────────────────────────
        if eliminar_outliers and feat_cols:
            antes     = len(df)
            df        = df[(np.abs(stats.zscore(df[feat_cols])) < 3).all(axis=1)]
            despues   = len(df)
            outliers_info = {
                "antes":      antes,
                "despues":    despues,
                "eliminados": antes - despues,
            }

        # ── Distribución de clases ────────────────────────────────────
        conteo = df["label"].value_counts()
        total  = len(df)

        clases_dict = {}
        for i, (clase, count) in enumerate(conteo.items()):
            clases_dict[str(clase)] = {
                "count": int(count),
                "pct":   round(count / total * 100, 1),
                "color": COLORES[i % len(COLORES)],
            }

        # ── Gráfica balance ───────────────────────────────────────────
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        colores_bar = [COLORES[i % len(COLORES)] for i in range(len(conteo))]
        bars = ax1.bar(conteo.index.astype(str), conteo.values,
                       color=colores_bar, edgecolor='white', linewidth=1.5, width=0.6)
        ax1.set_title("Balance de clases", fontsize=14, fontweight='bold', pad=15)
        ax1.set_xlabel("Clase", fontsize=11)
        ax1.set_ylabel("Número de muestras", fontsize=11)
        ax1.set_facecolor('#f8fafc')
        fig1.patch.set_facecolor('white')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.tick_params(axis='x', rotation=0)
        for bar, v in zip(bars, conteo.values):
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.5,
                     str(v), ha='center', fontweight='bold', fontsize=11)
        plt.tight_layout()

        buf1 = BytesIO()
        fig1.savefig(buf1, format='png', dpi=120, bbox_inches='tight')
        buf1.seek(0)
        img_balance = base64.b64encode(buf1.read()).decode('utf-8')
        plt.close(fig1)

        # ── Pairplot ──────────────────────────────────────────────────
        paleta    = {str(c): COLORES[i % len(COLORES)]
                     for i, c in enumerate(df["label"].unique())}
        df_plot        = df[feat_cols + ["label"]].copy()
        df_plot["label"] = df_plot["label"].astype(str)

        sns.set_style("whitegrid")
        pair_grid = sns.pairplot(
            df_plot,
            hue="label",
            vars=feat_cols,
            palette=paleta,
            plot_kws={"alpha": 0.6, "s": 20},
            diag_kind="kde"
        )
        titulo = "Dispersión entre características"
        if eliminar_outliers:
            titulo += " (sin outliers)"
        pair_grid.figure.suptitle(titulo, y=1.02, fontsize=13, fontweight='bold')

        buf2 = BytesIO()
        pair_grid.figure.savefig(buf2, format='png', dpi=110, bbox_inches='tight')
        buf2.seek(0)
        img_pairplot = base64.b64encode(buf2.read()).decode('utf-8')
        plt.close('all')

        resultado = {
            "clases":        clases_dict,
            "total":         total,
            "img_balance":   img_balance,
            "img_pairplot":  img_pairplot,
            "outliers_info": outliers_info,
        }

        return render_template("analisis.html",
                               resultado=resultado,
                               filename=archivo.filename,
                               opciones=opciones)

    except Exception as e:
        flash(f"Error al procesar el archivo: {e}")
        return render_template("analisis.html", resultado=None,
                               filename=None, opciones=opciones)