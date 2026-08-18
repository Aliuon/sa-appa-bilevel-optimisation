function p = std_normal_pdf(z)
% STD_NORMAL_PDF  Standard normal PDF, avoiding a Statistics Toolbox
% dependency on normpdf.
    p = exp(-0.5 * z .^ 2) / sqrt(2 * pi);
end
