# Diagnóstico 28-sep: saltos de pose ArUco al pasar por plano

- `diag_ippe.py` / `diag_ippe.json`: en la Pi, ambas soluciones IPPE + error por cuadro (con/sin subpíxel).
- `cap_corners.py` / `corners.json`: esquinas reales de 307 cuadros con los 4 tags (tableta EC1).
- `tilt_series.json`: serie de /platform_pose con el salto −15.6° → +11.4° → −15.5°.
- Causa: separación Y configurada 83 mm vs 85 mm real (ajustada por mínimos cuadrados con corners.json)
  → inclinación fantasma ±12.5° (cos θ = 83/85). Con la config vieja |θy| nunca bajó de 6.5°.
- Fix: geometría corregida + desambiguación temporal + LM + salvaguarda SQPnP (cam_ball_balancer_cpp).
