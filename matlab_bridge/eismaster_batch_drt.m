function eismaster_batch_drt(inputDir, outputDir, drtToolsDir, methodTag, drtType, lambdaValue, coeffValue, derivativeOrder, dataUsed, inductanceMode, shapeControl)
if nargin < 11
    error('Expected 11 input arguments.');
end

addpath(drtToolsDir);
inputFiles = dir(fullfile(inputDir, '*.txt'));
if isempty(inputFiles)
    error('No TXT files found in input directory: %s', inputDir);
end

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

h = DRTtools();
cleanupObj = onCleanup(@() close_hidden_figure(h));

handles = guidata(h);
handles = configure_handles(handles, methodTag, str2double(drtType), str2double(lambdaValue), str2double(coeffValue), derivativeOrder, dataUsed, str2double(inductanceMode), shapeControl);
guidata(h, handles);

summaryPath = fullfile(outputDir, 'matlab_drt_run_summary.txt');
fidSummary = fopen(summaryPath, 'wt');
fprintf(fidSummary, 'method\t%s\n', methodTag);
fprintf(fidSummary, 'drt_type\t%s\n', drtType);
fprintf(fidSummary, 'lambda\t%s\n', lambdaValue);
fprintf(fidSummary, 'coeff\t%s\n', coeffValue);

for k = 1:numel(inputFiles)
    inputPath = fullfile(inputFiles(k).folder, inputFiles(k).name);
    [~, baseName, ~] = fileparts(inputPath);
    data = dlmread(inputPath);
    data = normalize_input_data(data);

    handles = guidata(h);
    handles = configure_handles(handles, methodTag, str2double(drtType), str2double(lambdaValue), str2double(coeffValue), derivativeOrder, dataUsed, str2double(inductanceMode), shapeControl);
    handles = DRTtools('apply_loaded_data_to_handles', h, [], handles, data);
    handles = guidata(h);
    handles.method_tag = lower(methodTag);
    guidata(h, handles);
    handles = run_selected_method(h, handles, methodTag);
    handles = guidata(h);

    exportPath = fullfile(outputDir, [baseName, '_DRT.txt']);
    write_drt_export(exportPath, handles);
    fprintf(fidSummary, 'file\t%s\t%s\n', inputFiles(k).name, exportPath);
end

fclose(fidSummary);
end

function handles = configure_handles(handles, methodTag, drtType, lambdaValue, coeffValue, derivativeOrder, dataUsed, inductanceMode, shapeControl)
set(handles.DRT_type, 'Value', drtType);
if isfield(handles, 'value') && ishghandle(handles.value)
    set(handles.value, 'String', num2str(lambdaValue, '%.12g'));
end
if isfield(handles, 'derivative') && ishghandle(handles.derivative)
    set_popup_to_value(handles.derivative, derivativeOrder);
end
if isfield(handles, 'shape') && ishghandle(handles.shape)
    set_popup_to_value(handles.shape, shapeControl);
end
if isfield(handles, 'data_used_popup') && ishghandle(handles.data_used_popup)
    try
        set_popup_to_value(handles.data_used_popup, dataUsed);
    catch
    end
end
set(handles.inductance, 'Value', inductanceMode);

handles.lambda = lambdaValue;
handles.coeff = coeffValue;
handles.der_used = derivativeOrder;
handles.data_used = dataUsed;
handles.shape_control = shapeControl;
handles.method_tag = lower(methodTag);
handles.plot_type = get_selected_popup_string(handles.DRT_type);
handles = DRTtools('inductance_Callback', h_from_handles(handles), [], handles);
guidata(h_from_handles(handles), handles);
end

function h = h_from_handles(handles)
h = handles.output;
end

function handles = run_selected_method(h, handles, methodTag)
switch lower(methodTag)
    case 'simple'
        handles = DRTtools('regularization_button_Callback', h, [], handles);
    case 'credit'
        DRTtools('bayesian_button_Callback', h, [], handles);
        handles = guidata(h);
    case 'bht'
        DRTtools('BHT_button_Callback', h, [], handles);
        handles = guidata(h);
    case 'peak'
        handles = DRTtools('regularization_button_Callback', h, [], handles);
        handles = guidata(h);
        DRTtools('peak_analysis_Callback', h, [], handles);
        handles = guidata(h);
    otherwise
        error('Unsupported DRT method tag: %s', methodTag);
end
end

function data = normalize_input_data(data)
if isempty(data)
    return;
end

index = find(data(:,1) == 0);
data(index,:) = [];
if data(1,1) < data(end,1)
    data = fliplr(data')';
end
end

function write_drt_export(fullFileName, handles)
fid = fopen(fullFileName, 'wt');

    switch handles.method_tag
        case 'simple'
            col_freq = handles.freq_fine(:);
            col_tau  = 1./col_freq;
            col_gamma = handles.gamma_ridge_fine(:);
            col_g = col_gamma .* col_freq;
            fprintf(fid, '%s\t%e\n', 'L', handles.x_ridge(1));
            fprintf(fid, '%s\t%e\n', 'R', handles.x_ridge(2));
        switch get(handles.DRT_type, 'Value')
            case 1
                fprintf(fid, 'tau\tgamma(tau)\n');
                fprintf(fid, '%e\t%e\n', [col_tau, col_gamma]');
            case 2
                fprintf(fid, 'freq\tgamma(freq)\n');
                fprintf(fid, '%e\t%e\n', [col_freq, col_gamma]');
            case 3
                fprintf(fid, 'tau\tg(tau)\n');
                fprintf(fid, '%e\t%e\n', [col_tau, col_g]');
            case 4
                fprintf(fid, 'freq\tg(freq)\n');
                fprintf(fid, '%e\t%e\n', [col_freq, col_g]');
        end
    case 'credit'
        col_freq = handles.freq_fine(:);
        col_tau = 1./col_freq;
        col_gamma = handles.gamma_ridge_fine(:);
        col_mean = handles.gamma_mean_fine(:);
        col_upper = handles.upper_bound_fine(:);
        col_lower = handles.lower_bound_fine(:);
        fprintf(fid, '%s\t%e\n', 'L', handles.x_ridge(1));
        fprintf(fid, '%s\t%e\n', 'R', handles.x_ridge(2));
        switch get(handles.DRT_type, 'Value')
            case 1
                fprintf(fid, 'tau\tMAP gamma\tMean gamma\tUpperbound gamma\tLowerbound gamma\n');
                fprintf(fid, '%e\t%e\t%e\t%e\t%e\n', [col_tau(:), col_gamma(:), col_mean(:), col_upper(:), col_lower(:)]');
            case 2
                fprintf(fid, 'freq\tMAP gamma\tMean gamma\tUpperbound gamma\tLowerbound gamma\n');
                fprintf(fid, '%e\t%e\t%e\t%e\t%e\n', [col_freq(:), col_gamma(:), col_mean(:), col_upper(:), col_lower(:)]');
            case 3
                fprintf(fid, 'tau\tMAP g\tMean g\tUpperbound g\tLowerbound g\n');
                fprintf(fid, '%e\t%e\t%e\t%e\t%e\n', [col_tau(:), (col_gamma(:).*col_freq(:)), (col_mean(:).*col_freq(:)), (col_upper(:).*col_freq(:)), (col_lower(:).*col_freq(:))]');
            case 4
                fprintf(fid, 'freq\tMAP g\tMean g\tUpperbound g\tLowerbound g\n');
                fprintf(fid, '%e\t%e\t%e\t%e\t%e\n', [col_freq(:), (col_gamma(:).*col_freq(:)), (col_mean(:).*col_freq(:)), (col_upper(:).*col_freq(:)), (col_lower(:).*col_freq(:))]');
        end
    case 'BHT'
        col_freq = handles.freq_fine(:);
        col_tau = 1./col_freq;
        col_re = handles.gamma_mean_fine_re(:);
        col_im = handles.gamma_mean_fine_im(:);
        fprintf(fid, '%s\t%e\n', 'L', handles.mu_L_0);
        fprintf(fid, '%s\t%e\n', 'R', handles.mu_R_inf);
        switch get(handles.DRT_type, 'Value')
            case 1
                fprintf(fid, 'tau\tgamma_Re\tgamma_Im\n');
                fprintf(fid, '%e\t%e\t%e\n', [col_tau(:), col_re(:), col_im(:)]');
            case 2
                fprintf(fid, 'freq\tgamma_Re\tgamma_Im\n');
                fprintf(fid, '%e\t%e\t%e\n', [col_freq(:), col_re(:), col_im(:)]');
            case 3
                fprintf(fid, 'tau\tg_Re\tg_Im\n');
                fprintf(fid, '%e\t%e\t%e\n', [col_tau(:), (col_re(:).*col_freq(:)), (col_im(:).*col_freq(:))]');
            case 4
                fprintf(fid, 'freq\tg_Re\tg_Im\n');
                fprintf(fid, '%e\t%e\t%e\n', [col_freq(:), (col_re(:).*col_freq(:)), (col_im(:).*col_freq(:))]');
        end
    case 'peak'
        p_fit = handles.p_result;
        peak_height = p_fit(1, :);
        peak_log_tau_mu = p_fit(2, :);
        peak_sigma = p_fit(3, :);
        fprintf(fid, '%s\t%e\n', 'L', handles.x_ridge(1));
        fprintf(fid, '%s\t%e\n', 'R', handles.x_ridge(2));
        fprintf(fid, 'function\tpeak_height*exp(-1/2*(log_tau - peak_position)^2/peak_width^2)\n');
        fprintf(fid, 'peak number\tpeak height\tpeak position\tpeak width\n');
        for i = 1:handles.N_peak
            fprintf(fid, '%d\t%e\t%e\t%e\n', i, peak_height(i), peak_log_tau_mu(i), peak_sigma(i));
        end
end

fclose(fid);
end

function set_popup_to_value(handleObj, targetString)
items = get(handleObj, 'String');
if ischar(items)
    items = cellstr(items);
end
idx = find(strcmp(items, targetString), 1, 'first');
if isempty(idx)
    error('Value %s not found in popup menu.', targetString);
end
set(handleObj, 'Value', idx);
end

function value = get_selected_popup_string(handleObj)
items = get(handleObj, 'String');
if ischar(items)
    items = cellstr(items);
end
value = items{get(handleObj, 'Value')};
end

function close_hidden_figure(h)
if ishghandle(h)
    close(h);
end
end
