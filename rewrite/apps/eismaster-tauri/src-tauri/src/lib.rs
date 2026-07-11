mod commands;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            commands::parse_file,
            commands::parse_folder,
            commands::export_drt_workbook,
            commands::run_matlab_drt,
            commands::discover_drt_files,
            commands::export_workbook_from_drt_dir,
            commands::get_dev_resource_paths,
            commands::inspect_spectrum,
            commands::fit_spectrum,
            commands::fit_batch_folder,
            commands::fit_batch_paths,
            commands::export_batch_workbook,
            commands::export_batch_workbook_from_paths,
            commands::export_single_fit,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
