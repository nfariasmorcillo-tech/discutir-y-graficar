import streamlit as st
import sympy as sp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuración de la interfaz
st.set_page_config(page_title="Discusión de Relaciones Algebráicas", page_icon="📐", layout="wide")

st.title("📐 Laboratorio Didáctico: Discusión y Gráfica de Relaciones")
st.markdown("Analiza curvas analíticas paso a paso utilizando **álgebra clásica**, sin límites ni derivadas.")

# =========================================================================
# BARRA LATERAL: ENTRADA DE LA RELACIÓN
# =========================================================================
st.sidebar.header("📝 Entrada de la Ecuación")
st.sidebar.markdown("Ingresa la ecuación expresada como $E(x, y) = 0$")

# Ejemplos del archivo .tex para que pruebes la consistencia
ejemplo = st.sidebar.selectbox("Cargar un ejemplo del archivo de referencia:", [
    "Personalizado",
    "Parábola: x**2 + y - 2*x - 3",
    "Hipérbola: x**2 - y**2 - 4",
    "Racional R(x): y - (x + 1)/(x - 2)",
    "Racional R(x^2): y - (x**2 - 1)/(x**2 - 4)"
])

if ejemplo == "Parábola: x**2 + y - 2*x - 3":
    eq_def = "x**2 + y - 2*x - 3"
elif ejemplo == "Hipérbola: x**2 - y**2 - 4":
    eq_def = "x**2 - y**2 - 4"
elif ejemplo == "Racional R(x): y - (x + 1)/(x - 2)":
    eq_def = "y - (x + 1)/(x - 2)"
elif ejemplo == "Racional R(x^2): y - (x**2 - 1)/(x**2 - 4)":
    eq_def = "y - (x**2 - 1)/(x**2 - 4)"
else:
    eq_def = "x**2 + y - 2*x - 3"

ecuacion_str = st.sidebar.text_input("Expresión E(x, y):", value=eq_def)

st.sidebar.info("💡 **Recordatorio de Sintaxis:**\n"
                "• Multiplicación: `3*x` o `x*y`\n"
                "• Potencias: `x**2` o `y**2`\n"
                "• Divisiones: `(x+1)/(x-2)`")

calcular = st.sidebar.button("Discutir Relación", type="primary")

# =========================================================================
# PROCESAMIENTO MATEMÁTICO (MÉTODO DE DISCUSIÓN)
# =========================================================================
if calcular:
    try:
        x, y = sp.symbols('x y', real=True)
        E = sp.sympify(ecuacion_str)
        
        st.header("Análisis Formal de la Relación")
        st.latex(rf"E(x, y): \quad {sp.latex(E)} = 0")
        st.divider()

        # -----------------------------------------------------------------
        # PASO 1: EXTENSIÓN (DOMINIO Y RANGO)
        # -----------------------------------------------------------------
        st.subheader("1) Extensión de la Curva")
        col_dom, col_ran = st.columns(2)
        
        with col_dom:
            st.markdown("**a) Dominio (Valores permitidos para $x$):**")
            try:
                # Intentamos despejar 'y' para analizar restricciones reales de forma algebraica
                despeje_y = sp.solve(E, y)
                if despeje_y:
                    st.markdown("Despejando $y$ obtenemos la(s) expresión(es):")
                    for sol in despeje_y:
                        st.latex(rf"y = {sp.latex(sol)}")
                    
                    # Intentar hallar el dominio implícito en los reales
                    dom_conjunto = sp.continuous_domain(despeje_y[0], x, sp.S.Reals)
                    st.info(f"Análisis del campo real: $D_f = {sp.latex(dom_conjunto)}$")
                else:
                    st.warning("No se pudo despejar explícitamente $y$. Requiere análisis implícito.")
            except Exception:
                st.error("Análisis analítico del dominio complejo para esta expresión.")

        with col_ran:
            st.markdown("**b) Rango (Valores permitidos para $y$):**")
            try:
                # Intentamos despejar 'x' para evaluar restricciones en el eje Y
                despeje_x = sp.solve(E, x)
                if despeje_x:
                    st.markdown("Despejando $x$ obtenemos la(s) expresión(es):")
                    for sol in despeje_x:
                        st.latex(rf"x = {sp.latex(sol)}")
                    
                    ran_conjunto = sp.continuous_domain(despeje_x[0], y, sp.S.Reals)
                    st.info(f"Análisis del campo real: $R_f = {sp.latex(ran_conjunto)}$")
                else:
                    st.warning("No se pudo despejar explícitamente $x$. Requiere análisis implícito.")
            except Exception:
                st.error("Análisis analítico del rango complejo para esta expresión.")

        st.divider()

        # -----------------------------------------------------------------
        # PASO 2: INTERSECCIONES CON LOS EJES
        # -----------------------------------------------------------------
        st.subheader("2) Intersecciones con los Ejes Coordenados")
        col_int_x, col_int_y = st.columns(2)
        
        with col_int_x:
            st.markdown("**Con el Eje $X$ (Haciendo $y = 0$):**")
            E_y0 = E.subs(y, 0)
            st.latex(rf"{sp.latex(E_y0)} = 0")
            sol_x0 = sp.solve(E_y0, x)
            sol_x0_reales = [s for s in sol_x0 if s.is_real]
            
            if sol_x0_reales:
                for sx in sol_x0_reales:
                    st.success(rf"Intercepto detectado: $\left({sp.latex(sx)},\, 0\right)$")
            else:
                st.write("No existen intersecciones reales con el Eje $X$.")

        with col_int_y:
            st.markdown("**Con el Eje $Y$ (Haciendo $x = 0$):**")
            E_x0 = E.subs(x, 0)
            st.latex(rf"{sp.latex(E_x0)} = 0")
            sol_y0 = sp.solve(E_x0, y)
            sol_y0_reales = [s for s in sol_y0 if s.is_real]
            
            if sol_y0_reales:
                for sy in sol_y0_reales:
                    st.success(rf"Intercepto detectado: $\left(0,\, {sp.latex(sy)}\right)$")
            else:
                st.write("No existen intersecciones reales con el Eje $Y$.")

        st.divider()

        # -----------------------------------------------------------------
        # PASO 3: SIMETRÍAS
        # -----------------------------------------------------------------
        st.subheader("3) Análisis algebraico de Simetrías")
        
        # Simetría Eje X: E(x, y) == E(x, -y)
        E_ejeX = E.subs(y, -y)
        sim_X = sp.simplify(E - E_ejeX) == 0
        
        # Simetría Eje Y: E(x, y) == E(-x, y)
        E_ejeY = E.subs(x, -x)
        sim_Y = sp.simplify(E - E_ejeY) == 0
        
        # Simetría Origen: E(x, y) == E(-x, -y)
        E_origen = E.subs({x: -x, y: -y})
        sim_Ori = sp.simplify(E - E_origen) == 0
        
        col_sx, col_sy, col_so = st.columns(3)
        with col_sx:
            st.markdown("**Respecto al Eje $X$:**")
            st.write("¿$E(x, y) = E(x, -y)$?")
            st.info("✅ **Hay Simetría**" if sim_X else "❌ No presenta simetría")
            
        with col_sy:
            st.markdown("**Respecto al Eje $Y$:**")
            st.write("¿$E(x, y) = E(-x, y)$?")
            st.info("✅ **Hay Simetría**" if sim_Y else "❌ No presenta simetría")
            
        with col_so:
            st.markdown("**Respecto al Origen:**")
            st.write("¿$E(x, y) = E(-x, -y)$?")
            st.info("✅ **Hay Simetría**" if sim_Ori else "❌ No presenta simetría")

        st.divider()

        # -----------------------------------------------------------------
        # PASO 4: ASÍNTOTAS (MÉTODO ALGEBRAICO)
        # -----------------------------------------------------------------
        st.subheader("4) Determinación de Asíntotas Algebraicas")
        st.markdown("Determinadas mediante la anulación de los denominadores resultantes en los despejes algebraicos.")
        
        col_as_v, col_as_h = st.columns(2)
        
        with col_as_v:
            st.markdown("**Asíntotas Verticales:**")
            # Se obtienen buscando valores de x que rompen el denominador cuando despejamos y
            if despeje_y:
                denominadores_y = [sp.denom(sol) for sol in despeje_y if sp.denom(sol) != 1]
                if denominadores_y:
                    raices_v = sp.solve(denominadores_y[0], x)
                    if raices_v:
                        for rv in raices_v:
                            st.warning(rf"Asíntota Vertical en la recta: $x = {sp.latex(rv)}$")
                    else:
                        st.write("No se encontraron restricciones lineales divisorias.")
                else:
                    st.write("No posee denominadores en la variable $x$ (Estructura no racional).")
            else:
                st.write("Requiere análisis polinómico alternativo.")

        with col_as_h:
            st.markdown("**Asíntotas Horizontales:**")
            # Se obtienen buscando valores de y que rompen el denominador al despejar x
            if despeje_x:
                denominadores_x = [sp.denom(sol) for sol in despeje_x if sp.denom(sol) != 1]
                if denominadores_x:
                    raices_h = sp.solve(denominadores_x[0], y)
                    if raices_h:
                        for rh in raices_h:
                            st.warning(rf"Asíntota Horizontal en la recta: $y = {sp.latex(rh)}$")
                    else:
                        st.write("No se encontraron restricciones lineales divisorias.")
                else:
                    st.write("No posee denominadores en la variable $y$.")
            else:
                st.write("Estructura no analizable mediante despeje simple.")

        st.divider()

        # -----------------------------------------------------------------
        # PASO 5: TABULACIÓN Y REPRESENTACIÓN GRÁFICA (PGFPLOTS/MATPLOTLIB STYLE)
        # -----------------------------------------------------------------
        st.subheader("5) Tabulación Dinámica y Gráfica Coordinada")
        
        # Crear datos de muestra de manera segura evitando los puntos nulos
        f_num = sp.lambdify((x, y), E, "numpy")
        
        x_vals = np.linspace(-7, 7, 400)
        y_vals = np.linspace(-7, 7, 400)
        X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
        Z = f_num(X_grid, Y_grid)
        
        # Crear figura imitando las especificaciones limpias de TikZ de la hoja de problemas
        fig, ax = plt.subplots(figsize=(8, 7))
        
        # Dibujar curvas mediante contornos para tolerar relaciones implícitas (como circunferencias o hipérbolas)
        contour = ax.contour(X_grid, Y_grid, Z, levels=[0], colors='blue', linewidths=2)
        
        # Configurar los ejes cartesianos centrales
        ax.axhline(0, color='black', linewidth=1.2)
        ax.axvline(0, color='black', linewidth=1.2)
        
        # Rejilla fina como en el archivo .tex
        ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.6)
        
        ax.set_xlim([-6, 6])
        ax.set_ylim([-6, 6])
        ax.set_xlabel('$x$', fontsize=12, loc='right')
        ax.set_ylabel('$y$', fontsize=12, loc='top')
        ax.set_title(f"Gráfica de la Relación: {ecuacion_str} = 0", fontsize=12, fontweight='bold')
        
        # Pintar interceptos si existen para que se vean en la gráfica
        if sol_x0_reales:
            for sx in sol_x0_reales:
                ax.plot(float(sx), 0, 'ro', markersize=6, label="Intercepto X" if "Intercepto X" not in ax.get_legend_handles_labels()[1] else "")
        if sol_y0_reales:
            for sy in sol_y0_reales:
                ax.plot(0, float(sy), 'ro', markersize=6, label="Intercepto Y" if "Intercepto Y" not in ax.get_legend_handles_labels()[1] else "")

        # Graficar líneas guía de las asíntotas si se encontraron
        if despeje_y and denominadores_y and raices_v:
            for rv in raices_v:
                if rv.is_real:
                    ax.axvline(float(rv), color='red', linestyle=':', linewidth=1.5, label=f'Asín. Vert: x={rv}')
        if despeje_x and denominadores_x and raices_h:
            for rh in raices_h:
                if rh.is_real:
                    ax.axhline(float(rh), color='orange', linestyle=':', linewidth=1.5, label=f'Asín. Horiz: y={rh}')
        
        if len(ax.get_legend_handles_labels()[0]) > 0:
            ax.legend(loc='upper right', fontsize='small')

        # Mostrar gráfica y una tabla guía
        col_g, col_t = st.columns([3, 2])
        with col_g:
            st.pyplot(fig)
        
        with col_t:
            st.markdown("**Muestra de puntos tabulados sugeridos para graficar:**")
            if despeje_y:
                # Construir tabla evaluando valores enteros de x en la función principal
                valores_prueba_x = [-3, -2, -1, 0, 1, 2, 3, 4]
                filas_tabla = []
                for val in valores_prueba_x:
                    try:
                        res_y = despeje_y[0].subs(x, val).evalf()
                        if res_y.is_real:
                            filas_tabla.append({"Variable (x)": val, "Imagen (y)": f"{float(res_y):.3f}"})
                        else:
                            filas_tabla.append({"Variable (x)": val, "Imagen (y)": "No Definido 🚫"})
                    except:
                        continue
                st.dataframe(pd.DataFrame(filas_tabla), use_container_width=True, hide_index=True)
            else:
                st.write("Relación implícita compleja. Use los interceptos calculados en el paso 2 como anclajes de dibujo.")

    except Exception as e:
        st.error(f"Error procesando algebraicamente la relación: {e}. Asegúrate de ingresar una ecuación bien estructurada.")