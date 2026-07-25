"""
Genera un documento PDF de ejemplo: Política de Privacidad y Seguridad
de una fintech ficticia. Este archivo sirve únicamente como documento
de prueba para el agente. Reemplázalo por tus propios documentos reales
en /data.
"""
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

doc = SimpleDocTemplate("data/politica_privacidad.pdf", pagesize=letter,
                         topMargin=0.8*inch, bottomMargin=0.8*inch)
styles = getSampleStyleSheet()
h1 = styles["Heading1"]
h2 = styles["Heading2"]
body = ParagraphStyle("body", parent=styles["Normal"], spaceAfter=10, leading=15)

story = []

story.append(Paragraph("Política de Privacidad, Seguridad y Términos de Uso", styles["Title"]))
story.append(Paragraph("NelisaPay — Documento de ejemplo (ficticio)", styles["Normal"]))
story.append(Spacer(1, 20))

story.append(Paragraph("1. Política de Privacidad y Protección de Datos", h1))
story.append(Paragraph(
    "NelisaPay recopila datos personales como nombre completo, número de identificación, "
    "correo electrónico, número de teléfono y datos de la cuenta bancaria vinculada, "
    "únicamente con el fin de prestar el servicio de pagos y transferencias. "
    "Los datos se almacenan cifrados en reposo y en tránsito (AES-256 y TLS 1.3). "
    "NelisaPay no vende ni comparte datos personales con terceros con fines comerciales. "
    "Los datos pueden compartirse con autoridades competentes únicamente cuando exista "
    "un requerimiento legal válido. El usuario puede solicitar la eliminación de su cuenta "
    "y de sus datos personales escribiendo a privacidad@nelisapay.com, y la eliminación "
    "se procesará en un plazo máximo de 30 días hábiles.", body))

story.append(Paragraph("2. Términos y Condiciones de Uso", h1))
story.append(Paragraph(
    "El uso de la plataforma NelisaPay implica la aceptación de estos términos. "
    "El usuario debe ser mayor de 18 años y proporcionar información veraz al momento "
    "del registro. Está prohibido usar la plataforma para actividades ilícitas, lavado "
    "de dinero, financiamiento del terrorismo o fraude. NelisaPay se reserva el derecho "
    "de suspender o cerrar cuentas que incumplan estos términos, previa notificación al usuario "
    "cuando sea posible. Los términos pueden actualizarse periódicamente; los cambios "
    "relevantes se notificarán con al menos 15 días de anticipación.", body))

story.append(Paragraph("3. Preguntas Frecuentes sobre Transacciones y Límites", h1))
story.append(Paragraph(
    "El límite de transferencia diario para cuentas verificadas (KYC completo) es de "
    "$5,000 USD, y para cuentas no verificadas es de $500 USD. El límite mensual para "
    "cuentas verificadas es de $50,000 USD. Las transferencias entre cuentas NelisaPay "
    "son instantáneas. Las transferencias a bancos externos tardan entre 1 y 3 días hábiles. "
    "Si una transacción no se refleja después de 3 días hábiles, el usuario puede abrir "
    "un caso de soporte desde la sección 'Ayuda' de la aplicación.", body))

story.append(Paragraph("4. Política de Seguridad y Prevención de Fraudes", h1))
story.append(Paragraph(
    "NelisaPay utiliza autenticación de dos factores (2FA) obligatoria para inicios de "
    "sesión desde dispositivos nuevos y para transferencias mayores a $1,000 USD. "
    "Un sistema de monitoreo automatizado detecta patrones inusuales de transacciones "
    "(montos atípicos, ubicaciones geográficas inconsistentes, velocidad de transacciones) "
    "y puede congelar temporalmente una cuenta mientras se verifica la identidad del usuario. "
    "NelisaPay nunca solicita contraseñas o códigos de verificación por teléfono, correo "
    "o mensaje de texto. Cualquier comunicación que lo solicite debe considerarse un intento "
    "de fraude (phishing) y debe reportarse a seguridad@nelisapay.com.", body))

story.append(Paragraph("5. Tarifas y Comisiones del Servicio", h1))
story.append(Paragraph(
    "Las transferencias entre cuentas NelisaPay no tienen ningún costo. Las transferencias "
    "a cuentas bancarias externas tienen una comisión fija de $1.50 USD por transacción. "
    "Los retiros en cajeros automáticos afiliados tienen una comisión de $2.00 USD por retiro. "
    "El cambio de divisas dentro de la plataforma aplica una comisión del 1.5% sobre el monto "
    "convertido. No existen comisiones por mantenimiento de cuenta ni por inactividad.", body))

doc.build(story)
print("PDF generado en data/politica_privacidad.pdf")
