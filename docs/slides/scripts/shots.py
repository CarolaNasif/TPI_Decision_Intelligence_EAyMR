"""Capturas nítidas de los 3 módulos de la app local, con Edge vía Playwright.

Ventana angosta (560 px) a 3x: cada captura muestra un solo bloque con letra grande.
"""
import sys

from playwright.sync_api import sync_playwright

out = sys.argv[1]
URL = "http://localhost:8503"
W, H = 560, 1100


def shot(page, locator, name, height, pad_top=12):
    """Deja el elemento arriba de la ventana y recorta el bloque principal desde ahí."""
    locator.scroll_into_view_if_needed()
    page.wait_for_timeout(500)
    box = locator.bounding_box()
    page.evaluate(
        "(dy) => { const m = document.querySelector('[data-testid=\"stMain\"]') || document.scrollingElement;"
        " m.scrollBy(0, dy); }",
        box["y"] - 80,
    )
    page.wait_for_timeout(2500)  # da tiempo a que Plotly dibuje
    page.mouse.move(2, 2)
    page.wait_for_timeout(400)
    box = locator.bounding_box()
    main = page.locator('[data-testid="stMainBlockContainer"]').bounding_box()
    top = max(box["y"] - pad_top, 0)
    clip = {"x": main["x"] + 6, "y": top, "width": main["width"] - 12, "height": height}
    page.screenshot(path=f"{out}/{name}.png", clip=clip)
    print(name, clip)


with sync_playwright() as p:
    browser = p.chromium.launch(channel="msedge", headless=True)
    ctx = browser.new_context(viewport={"width": W, "height": H}, device_scale_factor=3, locale="es-AR")
    page = ctx.new_page()
    page.goto(URL, wait_until="networkidle")
    page.get_by_text("Diagnóstico y exploración").first.wait_for(timeout=60000)
    page.wait_for_timeout(3000)

    # ---- módulo 1: proyección PCA
    shot(page, page.get_by_text("Proyección PCA", exact=False).first, "app_m1", 640)
    shot(page, page.get_by_text("Distinguir Train / Test", exact=False).first, "app_m1_chart", 470)

    # ---- módulo 2: perfil riesgoso (Tenure mínimo + reclamo) y explicación por caso
    page.get_by_role("tab", name="2 · Simulador Predictivo (What-Is)").click()
    page.wait_for_timeout(1500)
    slider = page.get_by_role("slider").first  # Tenure
    slider.focus()
    page.keyboard.press("Home")
    page.wait_for_timeout(800)
    page.get_by_text("¿Tuvo un reclamo reciente?").first.click()
    page.wait_for_timeout(800)
    page.get_by_role("button", name="Calcular probabilidad de churn").click()
    h2 = page.get_by_text("¿Por qué esta probabilidad?", exact=False).first
    h2.wait_for(timeout=30000)
    page.wait_for_timeout(2000)
    shot(page, page.get_by_text("Probabilidad calibrada de churn").first, "app_m2_prob", 360)
    shot(page, h2, "app_m2", 430)

    # ---- módulo 3
    page.get_by_role("tab", name="3 · Simulador Causal (What-If)").click()
    page.wait_for_timeout(2000)
    shot(page, page.get_by_text("Semáforo de refutaciones", exact=False).first, "app_m3", 330)
    shot(page, page.get_by_text("Positividad / soporte común", exact=False).first, "app_m3_soporte", 330)
    shot(page, page.get_by_text("Curva de CATE por perfil", exact=False).first, "app_m3_cate", 760)
    browser.close()
print("ok")
