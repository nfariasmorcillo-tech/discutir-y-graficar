import streamlit as st
import sympy as sp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuración de la interfaz web
st.set_page_config(page_title="Discusión de Relaciones Algebráicas", page_icon="📐", layout="wide")

st.title("📐 Laboratorio Didáctico: Discusión y Gráfica de Relaciones")
st.markdown("Analiza curvas analíticas paso a paso utilizando **álgebra clásica**, sin límites ni derivadas.")

# =========================================================================
# BARRA LATERAL: ENTRADA DE LA RELACIÓN
# =========================================================================
st.sidebar.header("📝 Entrada de la Ecuación")
st.sidebar.markdown("Ingresa la ecuación expresada como $E(x, y) = 0$")

# Ejemplo por defecto basado en tu ejercicio de prueba
eq_def = "y - (2*x - 3)/(x**2 + x - 2)"

ecuacion_str = st.sidebar.text_input("Expresión E(x, y):", value=eq_def)

st.sidebar.info("💡 **Sintaxis matemática:**\n"
                "• Multiplicación: `2*x`\n"
                "• Potencias: `x**2`\n"
                "• División: `(2*x - 3)/(x**2 + x - 2)`")

calcular = st.sidebar.button("Discutir Relación", type="primary")

# =========================================================================
# MOTOR DE CÁLCULO ALGEBRAICO (MÉTODO DE DISCUSIÓN ALGEBRAICA)
# =========================================================================
if calcular:
    try:
        x, y = sp.symbols('x y', real=True)
        # Convertimos a expresión matemática pura de SymPy
        E_raw = sp.sympify(ecuacion_str)
        
        # Juntamos los términos sobre un común denominador si es una expresión racional
        E = sp.together(E_raw)
        
        st.header("Análisis Formal de la Relación")
        st.latex(rf"E(x, y): \quad {sp.latex(E_raw)} = 0")
        st.divider()

        # Separar el numerador y denominador general resultante
        num_E, den_E = sp.fraction(E)

        # -----------------------------------------------------------------
        # PASO 1: EXTENSIÓN (DOMINIO Y RANGO ALGEBRAICO)
        # -----------------------------------------------------------------
        st.subheader("1) Extensión de la Curva")
        col_dom, col_ran = st.columns(2)
        
        with col_dom:
            st.markdown("**a) Dominio (Valores permitidos para $x$):**")
            despeje_y = sp.solve(E_raw, y)
            if despeje_y:
                st.markdown("Expresión explícita de la relación:")
                st.latex(rf"y = {sp.latex(despeje_y[0])}")
                
                num_y, den_y = sp.fraction(despeje_y[0])
                if den_y != 1:
                    restricciones_x = sp.solve(den_y, x)
                    st.markdown("Restricción: El denominador no puede ser cero:")
                    st.latex(rf"{sp.latex(den_y)} \neq 0")
                    if restricciones_x:
                        # Corregido: Preparamos los elementos fuera de la f-string para evitar fallas de sintaxis
                        elementos_invalidos = ", ".join([sp.latex(r) for r in restricciones_x])
                        texto_dominio = rf"D_f = \mathbb{{R}} \setminus \{{ {elementos_invalidos} \}}"
                        st.info(rf"El dominio es todos los reales excepto: ${texto_dominio}$")
                else:
                    st.info("No hay restricciones en el denominador. $D_f: \mathbb{R}$")
            else:
                st.info("Análisis implícito del Dominio.")

        with col_ran:
            st.markdown("**b) Rango Algebraico (Criterio de la Ecuación Cuadrática):**")
            
            # Expandimos el numerador unificado para analizar el grado con respecto a x
            eq_polinomio = sp.expand(num_E)
            
            try:
                poly_x = sp.Poly(eq_polinomio, x)
                grado_x = poly_x.degree()
                
                if grado_x == 2:
                    A_y = poly_x.coeff_monomial(x**2)
                    B_y = poly_x.coeff_monomial(x)
                    C_y = poly_x.coeff_monomial(1)
                    
                    st.markdown("Reordenando como ecuación de segundo grado en términos de $x$:")
                    st.latex(rf"\left({sp.latex(A_y)}\right)x^2 + \left({sp.latex(B_y)}\right)x + \left({sp.latex(C_y)}\right) = 0")
                    st.markdown("Para que existan soluciones reales en $x$, exigimos que el discriminante sea no negativo ($\Delta \ge 0$):")
                    
                    discriminante = sp.simplify(B_y**2 - 4*A_y*C_y)
                    st.latex(rf"\Delta = {sp.latex(discriminante)} \ge 0")
                    
                    # Resolver la inecuación para encontrar las regiones válidas del rango de y
                    rango_sol = sp.solve(discriminante >= 0, y)
                    st.success(rf"Rango Analítico Calculado: $R_f = {sp.latex(rango_sol)}$")
                else:
                    # Intento de despeje directo para x si es de primer grado lineal con respecto a x
                    despeje_x = sp.solve(E_raw, x)
                    if despeje_x:
                        num_x, den_x = sp.fraction(despeje_x[0])
                        if den_x != 1:
                            restricciones_y = sp.solve(den_x, y)
                            elementos_invalidos_y = ", ".join([sp.latex(r) for r in restricciones_y])
                            st.success(rf"Rango Analítico Calculado: $R_f = \mathbb{{R}} \setminus \{{ {elementos_invalidos_y} \}}$")
                        else:
                            st.success("Rango Analítico: $\mathbb{R}$")
                    else:
                        st.info("Relación no lineal con respecto a $x$. Análisis analítico implícito.")
            except:
                st.info("No se pudo clasificar o despejar el rango usando métodos algebraicos estándar de segundo grado.")

        st.divider()

        # -----------------------------------------------------------------
        # PASO 2: INTERSECCIONES CON LOS EJES
        # -----------------------------------------------------------------
        st.subheader("2) Intersecciones con los Ejes Coordenados")
        col_int_x, col_int_y = st.columns(2)
        
        with col_int_x:
            st.markdown("**Con el Eje $X$ (Haciendo $y = 0$):**")
            E_y0 = E_raw.subs(y, 0)
            st.latex(rf"E(x, 0): \quad {sp.latex(E_y0)} = 0")
            sol_x0 = sp.solve(E_y0, x)
            sol_x0_reales = [s for s in sol_x0 if s.is_real]
            
            if sol_x0_reales:
                for sx in sol_x0_reales:
                    st.success(rf"Intercepto detectado: $\left({sp.latex(sx)},\, 0\right)$")
            else:
                st.write("No existen intersecciones reales con el Eje $X$.")

        with col_int_y:
            st.markdown("**Con el Eje $Y$ (Haciendo $x = 0$):**")
            E_x0 = E_raw.subs(x, 0)
            st.latex(rf"E(0, y): \quad {sp.latex(E_x0)} = 0")
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
        st.subheader("3) Análisis Algebraico de Simetrías")
        
        # Simetría Eje X: E(x, y) == E(x, -y)
        E_ejeX = E_raw.subs(y, -y)
        sim_X = sp.simplify(E_raw - E_ejeX) == 0
        
        # Simetría Eje Y: E(x, y) == E(-x, y)
        E_ejeY = E_raw.subs(x, -x)
        sim_Y = sp.simplify(E_raw - E_ejeY) == 0
        
        # Simetría Origen: E(x, y) == E(-x, -y)
        E_origen = E_raw.subs({x: -x, y: -y})
        sim_Ori = sp.simplify(E_raw - E_origen) == 0
        
        col_sx, col_sy, col_so = st.columns(3)
        with col_sx:
            st.markdown("**Respecto al Eje $X$:**")
            st.info("✅ **Hay Simetría**" if sim_X else "❌ No presenta simetría")
            
        with col_sy:
            st.markdown("**Respecto al Eje $Y$:**")
            st.info("✅ **Hay Simetría**" if sim_Y else "❌ No presenta simetría")
            
        with col_so:
            st.markdown("**Respecto al Origen:**")
            st.info("✅ **Hay Simetría**" if sim_Ori else "❌ No presenta simetría")

        st.divider()

        # -----------------------------------------------------------------
        # PASO 4: ASÍNTOTAS ALGEBRAICAS
        # -----------------------------------------------------------------
        st.subheader("4) Determinación de Asíntotas Algebraicas")
        col_as_v, col_as_h = st.columns(2)
        
        raices_v = []
        raices_h = []
        
        with col_as_v:
            st.markdown("**Asíntotas Verticales:**")
            if despeje_y:
                num_y, den_y = sp.fraction(despeje_y[0])
                if den_y != 1:
                    raices_v = sp.solve(den_y, x)
                    if raices_v:
                        for rv in raices_v:
                            st.warning(rf"Asíntota Vertical en la recta: $x = {sp.latex(rv)}$")
                    else:
                        st.write("No se encontraron raíces reales en el denominador.")
                else:
                    st.write("No posee denominadores en la variable $x$.")
            else:
                st.write("No se pudo analizar de forma analítica explícita.")

        with col_as_h:
            st.markdown("**Asíntotas Horizontales:**")
            despeje_x = sp.solve(E_raw, x)
            if despeje_x:
                num_x, den_x = sp.fraction(despeje_x[0])
                if den_x != 1:
                    raices_h = sp.solve(den_x, y)
                    if raices_h:
                        for rh in raices_h:
                            st.warning(rf"Asíntota Horizontal en la recta: $y = {sp.latex(rh)}$")
                    else:
                        st.write("No se encontraron raíces reales en el denominador.")
                else:
                    st.write("No posee denominadores en la variable $y$.")
            else:
                # Método analítico clásico de comparación de grados para funciones racionales
                if despeje_y:
                    num_y, den_y = sp.fraction(despeje_y[0])
                    try:
                        p_num = sp.Poly(num_y, x)
                        p_den = sp.Poly(den_y, x)
                        if p_num and p_den:
                            deg_num = p_num.degree()
                            deg_den = p_den.degree()
                            if deg_num < deg_den:
                                raices_h = [0]
                                st.warning(rf"Asíntota Horizontal en el eje: $y = 0$")
                            elif deg_num == deg_den:
                                lc_num = p_num.leading_coeff()
                                lc_den = p_den.leading_coeff()
                                rh_val = lc_num / lc_den
                                raices_h = [rh_val]
                                st.warning(rf"Asíntota Horizontal en la recta: $y = {sp.latex(rh_val)}$")
                    except:
                        st.write("Forma no polinomial estándar para cálculo de límites infinitos.")

        st.divider()

        # -----------------------------------------------------------------
        # PASO 5: GRÁFICA ESTILO PGFPLOTS (TIKZ)
        # -----------------------------------------------------------------
        st.subheader("5) Representación Gráfica Coordinada")
        
        # Creación de la función analítica numérica para el mapeo en el plano
        f_num = sp.lambdify((x, y), E_raw, "numpy")
        
        x_plot = np.linspace(-7, 7, 500)
        y_plot = np.linspace(-7, 7, 500)
        X_grid, Y_grid = np.meshgrid(x_plot, y_plot)
        
        # Desactivamos advertencias por división de ceros en puntos asintóticos
        with np.errstate(divide='ignore', invalid='ignore'):
            Z = f_num(X_grid, Y_grid)
        
        fig, ax = plt.subplots(figsize=(8, 7))
        
        # Dibujamos las curvas continuas donde el nivel de contorno algebraico sea igual a 0
        ax.contour(X_grid, Y_grid, Z, levels=[0], colors='blue', linewidths=2)
        
        # Ejes cartesianos principales destacados
        ax.axhline(0, color='black', linewidth=1.2)
        ax.axvline(0, color='black', linewidth=1.2)
        
        # Rejilla milimetrada fina cuadriculada, imitando los PDF generados en TeX
        ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        
        ax.set_xlim([-6, 6])
        ax.set_ylim([-6, 6])
        ax.set_xlabel('$x$', fontsize=12, loc='right')
        ax.set_ylabel('$y$', fontsize=12, loc='top')
        ax.set_title(f"Lugar Geométrico: {ecuacion_str} = 0", fontsize=11, fontweight='bold')
        
        # Graficar líneas guía verticales discontinuas para las Asíntotas Verticales
        for rv in raices_v:
            if rv.is_real:
                ax.axvline(float(rv), color='red', linestyle=':', linewidth=1.5, label=f'A.V. x={rv}')
        # Graficar líneas guía horizontales discontinuas para las Asíntotas Horizontales
        for rh in raices_h:
            try:
                ax.axhline(float(rh), color='orange', linestyle=':', linewidth=1.5, label=f'A.H. y={rh}')
            except:
                pass
                
        if len(ax.get_legend_handles_labels()[0]) > 0:
            ax.legend(loc='upper right')

        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error general en el procesamiento algebraico: {e}. Comprueba la correcta colocación de los paréntesis.")
