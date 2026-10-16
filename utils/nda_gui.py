import os
import platform
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class NDAGeneratorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Negative Data Augmentation (NDA) Studio")
        self.root.geometry("1700x1000")
        self.root.minsize(1300, 850)

        # File & Spectrum State
        self.current_filename = "Synthetic Base Spectrum"
        self.num_channels = 2048
        self.energy_axis = np.linspace(0, 3000, self.num_channels)
        self.real_spectrum = self._generate_synthetic_real_spectrum()
        self.corrupted_spectrum = self.real_spectrum.copy()

        # Dynamic Individual Spikes List: list of dicts {'pos_keV': float, 'amp': float}
        self.custom_spikes = []

        # Setup Modern Theme & Layout
        self._apply_modern_theme()
        self._setup_layout()
        self._update_spectrum()

    def _apply_modern_theme(self):
        """Configures dark mode palette and extra-large TTK widget styling."""
        self.BG_DARK = "#0F172A"       # Deep slate navy
        self.PANEL_BG = "#1E293B"      # Dark slate panel
        self.CARD_BG = "#334155"       # Card frame
        self.TEXT_MAIN = "#F8FAFC"     # Off-white
        self.TEXT_MUTED = "#CBD5E1"    # Muted light grey
        self.ACCENT_BLUE = "#38BDF8"   # Bright sky blue
        self.ACCENT_GREEN = "#34D399"  # Emerald green
        self.ACCENT_RED = "#F87171"    # Coral red

        self.root.configure(bg=self.BG_DARK)

        style = ttk.Style()
        style.theme_use("clam")

        # Global Defaults (Base font set to 15pt)
        style.configure(".", background=self.PANEL_BG, foreground=self.TEXT_MAIN, font=("Segoe UI", 15))
        style.configure("TFrame", background=self.PANEL_BG)

        # Header Styles
        style.configure("Header.TLabel", font=("Segoe UI", 28, "bold"), foreground=self.ACCENT_BLUE, background=self.PANEL_BG)
        style.configure("CardHeader.TLabel", font=("Segoe UI", 16, "bold"), foreground=self.TEXT_MAIN, background=self.CARD_BG)
        style.configure("Value.TLabel", font=("Segoe UI", 16, "bold"), foreground=self.ACCENT_BLUE, background=self.CARD_BG)
        style.configure("Muted.TLabel", font=("Segoe UI", 14, "bold"), foreground=self.TEXT_MUTED, background=self.CARD_BG)

        # Card Labelframes (Section titles 18pt bold)
        style.configure("Card.TLabelframe", background=self.CARD_BG, borderwidth=1, relief="solid")
        style.configure("Card.TLabelframe.Label", font=("Segoe UI", 18, "bold"), foreground=self.ACCENT_BLUE, background=self.PANEL_BG)

        # Buttons (15pt bold with padded click targets)
        style.configure("TButton", font=("Segoe UI", 15, "bold"), padding=12, background=self.CARD_BG, foreground=self.TEXT_MAIN, borderwidth=0)
        style.map("TButton", background=[("active", "#2563EB"), ("pressed", "#1D4ED8")])

        style.configure("Accent.TButton", background="#2563EB", foreground="#FFFFFF")
        style.map("Accent.TButton", background=[("active", "#1D4ED8")])

        style.configure("Success.TButton", background="#059669", foreground="#FFFFFF")
        style.map("Success.TButton", background=[("active", "#047857")])

        style.configure("Danger.TButton", background="#DC2626", foreground="#FFFFFF")
        style.map("Danger.TButton", background=[("active", "#B91C1C")])

        style.configure("Reset.TButton", background="#475569", foreground="#FFFFFF")
        style.map("Reset.TButton", background=[("active", "#64748B")])

        # Scales / Sliders
        style.configure("TScale", background=self.CARD_BG, troughcolor=self.PANEL_BG)

    def _generate_synthetic_real_spectrum(self):
        """Generates a physically realistic synthetic gamma spectrum."""
        E = self.energy_axis
        continuum = 5000 * np.exp(-E / 400) + 100 * np.exp(-E / 1500) + 5
        peaks = (
            2000 * np.exp(-((E - 511) ** 2) / (2 * (12**2))) +
            1200 * np.exp(-((E - 1173) ** 2) / (2 * (20**2))) +
            1000 * np.exp(-((E - 1332) ** 2) / (2 * (20**2)))
        )
        counts = continuum + peaks
        return np.random.poisson(counts).astype(np.float32)

    def _setup_layout(self):
        # Top Action Bar
        top_bar = ttk.Frame(self.root, padding="18")
        top_bar.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top_bar, text="NDA Spectrum Studio", font=("Segoe UI", 28, "bold"), foreground=self.ACCENT_BLUE).pack(side=tk.LEFT, padx=(0, 20))

        self.file_label = ttk.Label(top_bar, text=f"Active File: {self.current_filename}", font=("Segoe UI", 16, "italic"), foreground=self.ACCENT_GREEN)
        self.file_label.pack(side=tk.LEFT, padx=10)

        # Top Bar Action Buttons
        ttk.Button(top_bar, text="Load Spectrum (.txt / .npy)", command=self._load_file, style="Accent.TButton").pack(side=tk.RIGHT, padx=6)
        ttk.Button(top_bar, text="Generate New Base", command=self._reset_synthetic).pack(side=tk.RIGHT, padx=6)
        ttk.Button(top_bar, text="? Reset All Modifications", command=self._reset_all_modifications, style="Reset.TButton").pack(side=tk.RIGHT, padx=14)

        # Main Split Panel
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 14))

        # Left Controls Sidebar (Expanded Width: 620px for extra-large text)
        self.control_canvas = tk.Canvas(main_container, bg=self.PANEL_BG, highlightthickness=0, width=620)
        scrollbar = ttk.Scrollbar(main_container, orient="vertical", command=self.control_canvas.yview)

        self.scrollable_control_frame = ttk.Frame(self.control_canvas, padding="14")
        self.scrollable_control_frame.bind(
            "<Configure>", lambda e: self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all"))
        )

        self.control_canvas.create_window((0, 0), window=self.scrollable_control_frame, anchor="nw")
        self.control_canvas.configure(yscrollcommand=scrollbar.set)

        self.control_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
        scrollbar.pack(side=tk.LEFT, fill=tk.Y)

        # Enable Mouse Wheel Scrolling Over Left Panel
        self._enable_mousewheel_scrolling(self.control_canvas, self.scrollable_control_frame)

        # Right Panel: Plot Canvas
        plot_frame = ttk.Frame(main_container, padding="8")
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Dark Matplotlib Canvas
        plt.style.use("dark_background")
        self.fig, self.ax = plt.subplots(figsize=(10, 7), facecolor=self.PANEL_BG)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # --- CONTROL CARDS ---
        self.sliders = {}

        # CARD 1: Custom Peak & Spike Location Manager
        self._build_spike_manager_card(self.scrollable_control_frame)

        # CARD 2: Compton Continuum Stripping
        self._create_slider_card(self.scrollable_control_frame, "Compton Continuum Stripping", [
            ("gap_start", "Gap Start (keV)", 200, 2800, 1500, 10),
            ("gap_width", "Gap Width (keV)", 0, 1200, 0, 10),
        ])

        # CARD 3: High-Frequency Noise Glitch
        self._create_slider_card(self.scrollable_control_frame, "High-Freq Noise Glitch", [
            ("ripple_amp", "Ripple Amplitude", 0, 300, 0, 1),
            ("ripple_freq", "Frequency Factor", 1, 50, 10, 1),
        ])

        # CARD 4: Unphysical Slope Reversal
        self._create_slider_card(self.scrollable_control_frame, "Unphysical Slope Reversal", [
            ("slope_start", "Slope Start (keV)", 1000, 2800, 2000, 10),
            ("slope_factor", "Slope Factor", 0, 5, 0, 0.1),
        ])

        # Bottom Export Bar
        export_frame = ttk.Frame(self.scrollable_control_frame, padding="5")
        export_frame.pack(fill=tk.X, pady=24)
        ttk.Button(export_frame, text="Save Corrupted (.npy)", command=self._save_npy, style="Success.TButton").pack(fill=tk.X, pady=6)
        ttk.Button(export_frame, text="Save Plot (.png)", command=self._save_plot).pack(fill=tk.X, pady=6)

    def _enable_mousewheel_scrolling(self, canvas, frame):
        """Attaches cross-platform mouse wheel scroll events to canvas and child widgets."""
        def _on_mousewheel(event):
            if platform.system() == "Linux":
                if event.num == 4:
                    canvas.yview_scroll(-2, "units")
                elif event.num == 5:
                    canvas.yview_scroll(2, "units")
            elif platform.system() == "Darwin":
                canvas.yview_scroll(int(-1 * event.delta), "units")
            else: # Windows
                canvas.yview_scroll(int(-1 * (event.delta / 60)), "units")

        def _bind_to_all(widget):
            widget.bind("<MouseWheel>", _on_mousewheel)
            widget.bind("<Button-4>", _on_mousewheel)
            widget.bind("<Button-5>", _on_mousewheel)
            for child in widget.winfo_children():
                _bind_to_all(child)

        frame.bind("<Configure>", lambda e: _bind_to_all(frame))
        canvas.bind("<MouseWheel>", _on_mousewheel)
        canvas.bind("<Button-4>", _on_mousewheel)
        canvas.bind("<Button-5>", _on_mousewheel)

    def _build_spike_manager_card(self, parent):
        card = ttk.LabelFrame(parent, text="Individual Delta Spike Manager", style="Card.TLabelframe", padding="16")
        card.pack(fill=tk.X, pady=12)

        # Peak Position Controls
        ttk.Label(card, text="Spike Energy (keV):", style="CardHeader.TLabel").pack(anchor=tk.W)
        self.new_spike_energy_var = tk.DoubleVar(value=2500.0)
        self.spike_energy_label = ttk.Label(card, text="2500.0 keV", style="Value.TLabel")
        self.spike_energy_label.pack(anchor=tk.E)

        energy_scale = ttk.Scale(
            card, from_=50.0, to=2950.0, variable=self.new_spike_energy_var,
            command=lambda v: self.spike_energy_label.config(text=f"{float(v):.1f} keV")
        )
        energy_scale.pack(fill=tk.X, pady=(0, 12))

        # Peak Amplitude Controls
        ttk.Label(card, text="Spike Amplitude (Counts):", style="CardHeader.TLabel").pack(anchor=tk.W)
        self.new_spike_amp_var = tk.DoubleVar(value=500.0)
        self.spike_amp_label = ttk.Label(card, text="500.0", style="Value.TLabel")
        self.spike_amp_label.pack(anchor=tk.E)

        amp_scale = ttk.Scale(
            card, from_=10.0, to=10000.0, variable=self.new_spike_amp_var,
            command=lambda v: self.spike_amp_label.config(text=f"{float(v):.1f}")
        )
        amp_scale.pack(fill=tk.X, pady=(0, 14))

        # Button to Add Spike
        ttk.Button(card, text="+ Add Spike at Selected Energy", command=self._add_custom_spike, style="Accent.TButton").pack(fill=tk.X, pady=8)

        # Listbox to show added spikes
        ttk.Label(card, text="Active Custom Spikes:", style="Muted.TLabel").pack(anchor=tk.W, pady=(14, 4))

        self.spikes_listbox = tk.Listbox(
            card, height=4, bg=self.CARD_BG, fg=self.TEXT_MAIN,
            selectbackground="#2563EB", borderwidth=1, relief="solid",
            font=("Segoe UI", 14)
        )
        self.spikes_listbox.pack(fill=tk.X, pady=6)

        # Action buttons for listbox
        btn_frame = ttk.Frame(card)
        btn_frame.pack(fill=tk.X, pady=8)
        ttk.Button(btn_frame, text="Delete Selected", command=self._remove_selected_spike, style="Danger.TButton").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        ttk.Button(btn_frame, text="Clear All Spikes", command=self._clear_all_spikes).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(4, 0))

    def _create_slider_card(self, parent, title, slider_defs):
        card = ttk.LabelFrame(parent, text=title, style="Card.TLabelframe", padding="16")
        card.pack(fill=tk.X, pady=12)

        for key, label, min_val, max_val, default, step in slider_defs:
            lbl_val = ttk.Label(card, text=f"{default}", style="Value.TLabel")
            lbl_val.pack(anchor=tk.E)

            lbl_title = ttk.Label(card, text=f"{label}:", style="CardHeader.TLabel")
            lbl_title.pack(anchor=tk.W, pady=(0, 4))

            var = tk.DoubleVar(value=default)
            slider = ttk.Scale(
                card, from_=min_val, to=max_val, variable=var,
                command=lambda val, v_lbl=lbl_val, st=step: self._on_slider_change(v_lbl, val, st)
            )
            slider.pack(fill=tk.X, pady=(0, 14))
            self.sliders[key] = (var, default)

    def _on_slider_change(self, label_widget, val, step):
        formatted_val = f"{float(val):.1f}" if step < 1 else f"{int(float(val))}"
        label_widget.config(text=formatted_val)
        self._update_spectrum()

    def _add_custom_spike(self):
        energy = round(self.new_spike_energy_var.get(), 1)
        amp = round(self.new_spike_amp_var.get(), 1)

        self.custom_spikes.append({"pos_keV": energy, "amp": amp})
        self._refresh_spikes_listbox()
        self._update_spectrum()

    def _remove_selected_spike(self):
        selected_idx = self.spikes_listbox.curselection()
        if selected_idx:
            idx = selected_idx[0]
            del self.custom_spikes[idx]
            self._refresh_spikes_listbox()
            self._update_spectrum()

    def _clear_all_spikes(self):
        self.custom_spikes.clear()
        self._refresh_spikes_listbox()
        self._update_spectrum()

    def _refresh_spikes_listbox(self):
        self.spikes_listbox.delete(0, tk.END)
        for i, spike in enumerate(self.custom_spikes, start=1):
            self.spikes_listbox.insert(tk.END, f"Spike #{i}: Energy = {spike['pos_keV']} keV | Amp = {spike['amp']}")

    def _reset_all_modifications(self):
        for key, (var, default) in self.sliders.items():
            var.set(default)

        self.custom_spikes.clear()
        self._refresh_spikes_listbox()
        self._update_spectrum()

    def _update_spectrum(self):
        y_hat = self.real_spectrum.copy()
        L = len(y_hat)

        # 1. Apply Individual Custom Spikes
        for spike in self.custom_spikes:
            pos_keV = spike["pos_keV"]
            amp = spike["amp"]
            idx = int(np.clip((pos_keV / 3000.0) * L, 0, L - 1))
            y_hat[idx] += amp

        # 2. Compton Gap Stripping
        gap_start = self.sliders["gap_start"][0].get()
        gap_width = self.sliders["gap_width"][0].get()
        if gap_width > 0:
            idx_start = int((gap_start / 3000.0) * L)
            idx_end = int(((gap_start + gap_width) / 3000.0) * L)
            y_hat[idx_start:idx_end] *= 0.01

        # 3. High-Frequency Ripple
        ripple_amp = self.sliders["ripple_amp"][0].get()
        ripple_freq = self.sliders["ripple_freq"][0].get()
        if ripple_amp > 0:
            channels = np.arange(L)
            ripple = ripple_amp * np.sin(2 * np.pi * ripple_freq * channels / L)
            y_hat = np.maximum(0, y_hat + ripple)

        # 4. Slope Reversal
        slope_start = self.sliders["slope_start"][0].get()
        slope_factor = self.sliders["slope_factor"][0].get()
        if slope_factor > 0:
            idx_slope = int((slope_start / 3000.0) * L)
            ramp = np.linspace(1.0, 1.0 + slope_factor, L - idx_slope)
            y_hat[idx_slope:] *= ramp

        self.corrupted_spectrum = y_hat
        self._plot()

    def _plot(self):
        self.ax.clear()

        # Dark theme graph styling with extra-large fonts
        self.ax.set_facecolor("#0F172A")
        self.ax.plot(self.energy_axis, self.real_spectrum, label="Real Spectrum (y)", color="#38BDF8", alpha=0.75, linewidth=2.0)
        self.ax.plot(self.energy_axis, self.corrupted_spectrum, label="Unphysical Corrupted (y_hat)", color="#F43F5E", linestyle="--", linewidth=2.0)

        self.ax.set_yscale("log")
        self.ax.set_ylim(bottom=1e-1)
        self.ax.set_xlabel("Energy (keV)", color="#CBD5E1", fontsize=18, fontweight="bold")
        self.ax.set_ylabel("Counts (Log Scale)", color="#CBD5E1", fontsize=18, fontweight="bold")
        self.ax.set_title(f"Target NDA Preview | File: {self.current_filename}", fontsize=20, color="#F8FAFC", fontweight="bold", pad=16)

        self.ax.tick_params(colors="#CBD5E1", labelsize=15)
        self.ax.grid(True, which="both", ls="--", alpha=0.25, color="#475569")
        self.ax.legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="#F8FAFC", fontsize=16, loc="upper right")

        self.fig.tight_layout()
        self.canvas.draw()

    def _parse_txt_file(self, file_path):
        try:
            data = np.loadtxt(file_path)
            if data.ndim == 2:
                data = data[:, -1]
            return data.astype(np.float32)
        except Exception:
            counts = []
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        try:
                            counts.append(float(parts[-1]))
                        except ValueError:
                            continue
            if not counts:
                raise ValueError("Could not parse spectrum numbers.")
            return np.array(counts, dtype=np.float32)

    def _load_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Supported Spectrum Files", "*.txt *.npy"),
                ("Text Spectra (*.txt)", "*.txt"),
                ("Numpy Binary (*.npy)", "*.npy"),
            ]
        )
        if not file_path:
            return

        try:
            ext = os.path.splitext(file_path)[1].lower()
            if ext == ".npy":
                data = np.load(file_path)
                if data.ndim > 1:
                    data = data[0]
            else:
                data = self._parse_txt_file(file_path)

            self.real_spectrum = data
            self.num_channels = len(data)
            self.energy_axis = np.linspace(0, 3000, self.num_channels)
            self.current_filename = os.path.basename(file_path)
            self.file_label.config(text=f"Active File: {self.current_filename}")
            self._update_spectrum()

        except Exception as e:
            messagebox.showerror("Error Loading File", f"Failed to load file:\n{e}")

    def _reset_synthetic(self):
        self.current_filename = "Synthetic Base Spectrum"
        self.file_label.config(text=f"Active File: {self.current_filename}")
        self.num_channels = 2048
        self.energy_axis = np.linspace(0, 3000, self.num_channels)
        self.real_spectrum = self._generate_synthetic_real_spectrum()
        self._update_spectrum()

    def _save_npy(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".npy", filetypes=[("Numpy Binary", "*.npy")])
        if file_path:
            np.save(file_path, self.corrupted_spectrum)

    def _save_plot(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image", "*.png")])
        if file_path:
            self.fig.savefig(file_path, dpi=300)


if __name__ == "__main__":
    root = tk.Tk()
    app = NDAGeneratorGUI(root)
    root.mainloop()