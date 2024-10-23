# Sample python snippet for tcviewer.py for exec custom code (F8)
# Uses global deterministic models to save tracks for Caribbean region for last several days

screenshot_prefix = 'det_genesis_caribbean'
custom_region='Caribbean'
# Days of model cycles
n_days = 3

total_model_cycles = n_days*4

# Switch to GENESIS
cls.switch_mode(mode="GENESIS")
# Switch to GLOBAL-DET
cls.genesis_selected_combobox.current(0)
cls.genesis_previous_selected = cls.genesis_selected.get()

# last n_days of model cycles
cls.latest_genesis_cycle()
current_time = datetime_utcnow().strftime("%Y-%m-%d-%H-%M-%S")
filenames = []
for i in range(total_model_cycles):
    cls.zoom_in(extents=custom_extents[custom_region])

    # Get label texts
    genesis_mode_text = cls.label_genesis_mode.cget('text')
    genesis_models_text = cls.genesis_models_label.cget('text')

    # Add overlays
    cls.fig.text(0.01, 0.98, genesis_mode_text, ha='left', va='top',
                 fontsize=12, fontweight='bold', color='yellow',
                 bbox=dict(facecolor='black', alpha=0.5))
    cls.fig.text(0.01, 0.95, genesis_models_text, ha='left', va='top',
                 fontsize=12, fontweight='bold', color='yellow',
                 bbox=dict(facecolor='black', alpha=0.5))

    # Wait for everything to finish before screenshot
    cls.root.update_idletasks()
    # prefix zeros for img number
    prefix_format = (total_model_cycles // 10) + 1
    filename = f"screenshots/{screenshot_prefix}-{current_time}-img{i+1:0{prefix_format}d}.png"
    cls.fig.savefig(filename)

    # Print the file name so later we can use the list to create an animation
    filenames.append(filename)
    cls.prev_genesis_cycle()

print(filenames)
