"""Génère les documents PDF de démonstration d'une entreprise FICTIVE (« Atlas Industrie »).

Les documents internes réels sont confidentiels : ces politiques RH et informatiques
inventées servent de corpus de test. Chaque chiffre précis (plafonds, délais...) est
repris dans eval/questions.json pour mesurer la qualité des réponses.

Usage : python scripts/make_sample_docs.py  ->  data/docs/*.pdf
"""

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

OUT = Path(__file__).resolve().parents[1] / "data" / "docs"
FOOTER = "Atlas Industrie (entreprise fictive) – document de démonstration du projet DocuMind"

DOCUMENTS = {
    "politique_teletravail.pdf": ("Politique de télétravail", [
        ("1. Objet et bénéficiaires",
         "La présente politique encadre le télétravail au sein d'Atlas Industrie. Elle s'applique aux salariés en CDI "
         "ayant validé leur période d'essai. Le télétravail n'est pas ouvert pendant la période d'essai, ni aux postes "
         "dont l'activité exige une présence physique sur le site de production."),
        ("2. Nombre de jours autorisés",
         "Le salarié peut télétravailler jusqu'à deux jours par semaine, sous réserve de l'accord de son manager. "
         "Les jours de télétravail ne peuvent pas être reportés d'une semaine sur l'autre. Le mercredi est un jour "
         "de présence obligatoire pour toutes les équipes afin de faciliter les réunions collectives."),
        ("3. Procédure de demande",
         "La demande est déposée sur le portail RH au moins dix jours avant la date de début souhaitée. Le manager "
         "dispose de cinq jours ouvrés pour répondre. En cas de refus, celui-ci doit être motivé par écrit."),
        ("PAGE", ""),
        ("4. Équipement et indemnité",
         "L'entreprise fournit un ordinateur portable, un casque audio et un accès VPN. Une indemnité forfaitaire de "
         "30 dirhams par jour télétravaillé est versée chaque mois avec le salaire pour couvrir les frais d'internet "
         "et d'électricité. Le matériel reste la propriété d'Atlas Industrie et doit être restitué en fin de contrat."),
        ("5. Horaires et joignabilité",
         "Pendant les jours de télétravail, le salarié doit être joignable entre 9 h 30 et 16 h 30, sur Teams et par "
         "téléphone. Le droit à la déconnexion s'applique en dehors de ces plages horaires."),
        ("6. Sécurité",
         "La connexion au système d'information se fait obligatoirement par le VPN de l'entreprise. Le travail depuis "
         "un lieu public (café, gare) n'est autorisé qu'avec un filtre de confidentialité sur l'écran."),
    ]),
    "procedure_notes_de_frais.pdf": ("Procédure de remboursement des notes de frais", [
        ("1. Principe",
         "Les dépenses engagées pour le compte de l'entreprise lors de déplacements professionnels sont remboursées "
         "sur présentation des justificatifs originaux. Toute dépense doit être raisonnable et justifiée par l'activité."),
        ("2. Délai de déclaration",
         "La note de frais est saisie dans l'application Finance au plus tard 30 jours après la date de la dépense. "
         "Au-delà de ce délai, la dépense n'est plus remboursée, sauf accord exceptionnel du directeur financier."),
        ("3. Plafonds de remboursement",
         "Les plafonds suivants s'appliquent par personne : repas, 150 dirhams ; nuit d'hôtel, 900 dirhams à Casablanca "
         "et Rabat et 700 dirhams dans les autres villes ; utilisation du véhicule personnel, 3 dirhams par kilomètre. "
         "Les déplacements en train se font en deuxième classe ; la première classe est autorisée pour les trajets de "
         "plus de quatre heures."),
        ("PAGE", ""),
        ("4. Circuit de validation",
         "La note de frais est d'abord validée par le manager direct, puis contrôlée par le service comptabilité. "
         "Les notes de frais d'un montant supérieur à 5 000 dirhams nécessitent en plus la validation du directeur "
         "financier."),
        ("5. Délai de remboursement",
         "Une fois validée, la note de frais est remboursée par virement sous 15 jours. Les avances sur frais peuvent "
         "être demandées pour les missions de plus de trois jours, dans la limite de 3 000 dirhams."),
        ("6. Dépenses exclues",
         "Ne sont pas remboursés : les amendes et contraventions, l'alcool, les dépenses personnelles et les "
         "surclassements non autorisés."),
    ]),
    "charte_informatique.pdf": ("Charte d'utilisation des moyens informatiques", [
        ("1. Mots de passe",
         "Chaque mot de passe comporte au moins 12 caractères, dont une majuscule, un chiffre et un caractère spécial. "
         "Il doit être changé tous les 90 jours et ne jamais être communiqué, y compris au support informatique. "
         "Après cinq tentatives erronées, le compte est verrouillé pendant 30 minutes."),
        ("2. Authentification forte",
         "L'authentification à deux facteurs (MFA) est obligatoire pour l'accès au VPN, à la messagerie depuis "
         "l'extérieur et aux applications financières. Le code est généré par l'application Microsoft Authenticator."),
        ("3. Postes de travail",
         "La session doit être verrouillée à chaque absence du poste ; un verrouillage automatique intervient après "
         "5 minutes d'inactivité. L'installation de logiciels est réservée au service informatique."),
        ("PAGE", ""),
        ("4. Supports amovibles",
         "L'utilisation de clés USB personnelles est interdite. Seules les clés chiffrées fournies par le service "
         "informatique sont autorisées pour échanger des fichiers avec des partenaires."),
        ("5. Messagerie et hameçonnage",
         "Tout courriel suspect (lien inhabituel, demande urgente de paiement, pièce jointe inattendue) doit être "
         "signalé avec le bouton « Signaler un hameçonnage » d'Outlook, sans cliquer sur les liens."),
        ("6. Signalement des incidents",
         "Tout incident de sécurité (perte de matériel, virus, fuite de données) est signalé immédiatement au support "
         "informatique au poste 4444 ou à l'adresse securite@atlas-industrie.example. La perte d'un ordinateur doit "
         "être déclarée dans les 24 heures."),
    ]),
    "politique_conges.pdf": ("Politique de congés et absences", [
        ("1. Congés payés annuels",
         "Chaque salarié acquiert un jour et demi de congé payé par mois de travail effectif, soit 18 jours ouvrables "
         "par an. Cette durée est augmentée d'un jour et demi par période de cinq ans d'ancienneté."),
        ("2. Demande de congé",
         "Les congés sont demandés sur le portail RH au moins 15 jours à l'avance. Pour une absence de plus de deux "
         "semaines, le délai est porté à un mois. Les congés d'été sont planifiés avant le 30 avril de chaque année."),
        ("3. Report des congés",
         "Les congés non pris peuvent être reportés sur l'année suivante dans la limite de 10 jours. Les jours "
         "reportés doivent être utilisés avant le 31 mars ; à défaut, ils sont perdus."),
        ("PAGE", ""),
        ("4. Congés pour événements familiaux",
         "Des jours d'absence rémunérés sont accordés : 4 jours pour le mariage du salarié, 3 jours pour une naissance, "
         "3 jours pour le décès d'un conjoint, d'un enfant ou d'un parent, et 2 jours pour le mariage d'un enfant."),
        ("5. Absence pour maladie",
         "Le salarié prévient son manager le jour même et transmet un certificat médical aux RH dans un délai de "
         "48 heures."),
    ]),
    "guide_accueil.pdf": ("Guide d'accueil des nouveaux collaborateurs", [
        ("1. Premier jour",
         "Le nouveau collaborateur est accueilli à 9 heures à la réception du siège, à Oujda, par son manager et un "
         "membre de l'équipe RH. Il reçoit son badge d'accès, son ordinateur et ses identifiants."),
        ("2. Horaires",
         "Les horaires de référence sont de 8 h 30 à 17 h 30 du lundi au vendredi, avec une pause déjeuner d'une heure. "
         "Une plage de souplesse permet d'arriver entre 8 h et 9 h 30."),
        ("3. Restauration et parking",
         "Le restaurant d'entreprise est ouvert de 12 h à 14 h ; le repas est subventionné à hauteur de 50 %. "
         "Le parking du personnel est accessible avec le badge ; les places proches de l'entrée sont réservées au "
         "covoiturage."),
        ("PAGE", ""),
        ("4. Parcours d'intégration",
         "Un parrain est désigné pour chaque nouvel arrivant pendant ses trois premiers mois. Un entretien de suivi "
         "est organisé avec le manager à la fin du premier mois, puis à la fin de la période d'essai."),
        ("5. Sécurité sur le site de production",
         "Le port des équipements de protection individuelle (chaussures de sécurité, gilet haute visibilité et "
         "lunettes) est obligatoire dans les ateliers. Une formation sécurité de deux heures est suivie la première "
         "semaine."),
    ]),
}


def build(filename: str, title: str, sections: list[tuple[str, str]]) -> None:
    styles = getSampleStyleSheet()

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.drawString(2 * cm, 1.2 * cm, FOOTER)
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
        canvas.restoreState()

    story = [Paragraph(title, styles["Title"]), Paragraph("Atlas Industrie – Version 2026", styles["Italic"]), Spacer(1, 18)]
    for heading, text in sections:
        if heading == "PAGE":
            story.append(PageBreak())
            continue
        story += [Paragraph(heading, styles["Heading2"]), Paragraph(text, styles["BodyText"]), Spacer(1, 8)]

    SimpleDocTemplate(str(OUT / filename), pagesize=A4, title=title, author="Atlas Industrie (fictive)",
                      topMargin=2 * cm, bottomMargin=2 * cm).build(story, onFirstPage=footer, onLaterPages=footer)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for filename, (title, sections) in DOCUMENTS.items():
        build(filename, title, sections)
        print(f"  {filename}")
    print(f"{len(DOCUMENTS)} documents générés dans {OUT}")


if __name__ == "__main__":
    main()
