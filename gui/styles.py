# gui/styles.py
# Farbschema und Schriftarten für tkinter GUI.

COLORS = {
    "bg_dark":       "#1a1d23",
    "bg_panel":      "#22262e",
    "bg_input":      "#2a2f3a",
    "accent":        "#16a085",
    "accent_hover":  "#1abc9c",
    "text_primary":  "#ecf0f1",
    "text_secondary":"#95a5a6",
    "text_muted":    "#7f8c8d",
    "success":       "#27ae60",
    "warning":       "#e67e22",
    "error":         "#e74c3c",
    "info":          "#3498db",
    "border":        "#3d4452",
    "btn_primary":   "#16a085",
    "btn_danger":    "#c0392b",
    "btn_neutral":   "#2c3e50",
}

FONTS = {
    "default":    ("Helvetica Neue", 11),
    "small":      ("Helvetica Neue", 9),
    "large":      ("Helvetica Neue", 13),
    "title":      ("Helvetica Neue", 16, "bold"),
    "mono":       ("Menlo", 10),
    "label":      ("Helvetica Neue", 10),
    "label_bold": ("Helvetica Neue", 10, "bold"),
}

PADDING = {
    "xs": 4,
    "sm": 8,
    "md": 14,
    "lg": 22,
}


def apply_dark_theme(root) -> None:
    """Setzt dunklen Hintergrund für root-Widget und ttk-Styles."""
    root.configure(bg=COLORS["bg_dark"])
    try:
        import tkinter.ttk as ttk
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TFrame", background=COLORS["bg_panel"])
        style.configure("TLabel",
                        background=COLORS["bg_panel"],
                        foreground=COLORS["text_primary"],
                        font=FONTS["default"])
        style.configure("TButton",
                        background=COLORS["btn_primary"],
                        foreground=COLORS["text_primary"],
                        font=FONTS["label_bold"],
                        borderwidth=0,
                        relief="flat")
        style.map("TButton",
                  background=[("active", COLORS["accent_hover"]),
                               ("pressed", COLORS["accent"])])
        style.configure("Danger.TButton",
                        background=COLORS["btn_danger"],
                        foreground=COLORS["text_primary"])
        style.configure("TProgressbar",
                        background=COLORS["accent"],
                        troughcolor=COLORS["bg_input"],
                        borderwidth=0,
                        thickness=8)
        style.configure("TNotebook",
                        background=COLORS["bg_panel"],
                        borderwidth=0)
        style.configure("TNotebook.Tab",
                        background=COLORS["bg_input"],
                        foreground=COLORS["text_secondary"],
                        padding=[12, 5],
                        font=FONTS["label"])
        style.map("TNotebook.Tab",
                  background=[("selected", COLORS["bg_panel"])],
                  foreground=[("selected", COLORS["accent"])])
        style.configure("TCheckbutton",
                        background=COLORS["bg_panel"],
                        foreground=COLORS["text_primary"],
                        font=FONTS["label"])
        style.configure("TRadiobutton",
                        background=COLORS["bg_panel"],
                        foreground=COLORS["text_primary"],
                        font=FONTS["label"])
        style.configure("TScale",
                        background=COLORS["bg_panel"],
                        troughcolor=COLORS["bg_input"])
        style.configure("TCombobox",
                        background=COLORS["bg_input"],
                        foreground=COLORS["text_primary"],
                        fieldbackground=COLORS["bg_input"],
                        selectbackground=COLORS["accent"])
        style.configure("TEntry",
                        background=COLORS["bg_input"],
                        foreground=COLORS["text_primary"],
                        fieldbackground=COLORS["bg_input"],
                        insertcolor=COLORS["text_primary"])
        style.configure("TSpinbox",
                        background=COLORS["bg_input"],
                        foreground=COLORS["text_primary"],
                        fieldbackground=COLORS["bg_input"])
        style.configure("TSeparator",
                        background=COLORS["border"])
    except Exception:
        pass
