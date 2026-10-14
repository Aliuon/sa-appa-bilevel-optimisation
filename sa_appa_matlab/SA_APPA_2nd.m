%% SA_APPA_2nd.m -- single-file MATLAB translation of the SA-APPA experiment codebase
%
% PURPOSE
%   Self-contained translation of the Python codebase in sa_appa_codebase.zip
%   (sa_appa_experiments/ + sa_appa_figures/) that produced the signed-off
%   Scenario 1-7 results and the Appendix B/D tables: the 44-instance
%   benchmark suite, all 13 methods (M1-M13), the Friedman/Conover/Nemenyi
%   statistics, the Appendix B tables and the figure set. Run it as the bare
%   command
%
%       >> SA_APPA_2nd
%
%   with no input arguments. This is a SERIAL, straightforward-loop
%   translation (a plain per-individual loop wherever Python has one). A
%   vectorised + parallel sibling is a separate file (SA_APPA_3rd.m).
%
% SOURCE OF TRUTH
%   The canonical, locked implementation that produced the signed-off results:
%   the ORIGINAL, UNTUNED M1 configuration -- p_f held at 0.6, beta/gamma/delta
%   at the hardcoded constants 0.2/0.3/0.3 -- exactly as it ran, INCLUDING the
%   documented Eq. 11 / p_f implementation gap (the manuscript specifies a p_f
%   update, Eq. 11; the code that produced the results never varied p_f).
%   Nothing here silently "fixes" the mechanism to match the paper's equations;
%   doing so would make MATLAB and Python two different algorithms.
%   The pf_eta-corrected variant (Appendix D.7 diagnostic) is translated as a
%   separate, clearly labelled local function, runM1_AppendixD7Diagnostic(),
%   which the default sweep never calls.
%
% M5 / M6 LABELS (resolved)
%   The code that ran (methods.py METHOD_SPECS, config.py) defines M5 = DE and
%   M6 = PSO, and this file follows it. The older build_appendix_b.py label
%   swap ("M5 (PSO engine)", "M6 (DE engine)") was corrected in the
%   Python codebase and manuscript (October 2026).
%
% SCENARIO NAMES AND M10-M13
%   Scenario names follow the manuscript (scenarios.py): initial design, GA
%   hybridisation, search engine, constraint handling, step-size rule, in-house
%   surrogate-assisted engine baselines, fully nested baseline. M10-M13 are
%   in-house baselines; M12 and M13 are the same nested GA with different seed
%   streams. The manuscript's M14 (BL-SAEA re-implementation) and Scenario 8
%   exist only in the Python code (methods_m14.py) and are not translated here.
%
% REQUIREMENTS
%   MATLAB R2016b or later (local functions in scripts, implicit expansion).
%   GNU Octave is NOT supported: it has no RandStream class, which this file
%   uses for independent per-population random streams.
%   NO add-on toolboxes are required. Everything that would normally come from
%   the Statistics and Machine Learning Toolbox is hand-rolled from base
%   MATLAB: GP regression (chol), differential evolution, Latin hypercube
%   sampling, normal pdf/cdf (erf/erfc), F / t / studentized-range quantiles
%   (betainc, integral, bisection), boxplots (patch/line) and percentiles.
%   The local-polish step inside the lower-level solver is a hand-written
%   bound-constrained L-BFGS (see LOWER-LEVEL SOLVER NOTE). The Java VM is used only for the MD5 seed
%   rule (java.security.MessageDigest); without it a fallback hash is used.
%
% EXECUTION CAVEAT
%   This file has NOT been run in MATLAB (none was available to the author).
%   It was run under GNU Octave 8 through a test-only copy in which RandStream
%   was replaced by a small shim: the full QUICK_TEST pipeline (sweep, tables,
%   statistics, figures, validation report) completed. Against the Python code
%   the following were checked numerically: the MD5 seed rule (identical), all
%   44 benchmark instances (F, f, G, g, bounds, dimensions agree to 1e-12), the
%   GP / RBF surrogates and the EI / CEI / penalised-EI acquisitions (1e-12),
%   and the Friedman / Conover / Nemenyi statistics recomputed from Python's own
%   raw summary.csv (1e-11). The 13 methods were compared with Python only
%   statistically (10 runs on TP1 and TP5): same order of magnitude and same
%   method ordering. The full 17,160-run sweep has not been run. MATLAB-only
%   graphics calls (legend options, print) were not exercised in MATLAB itself.
%   Run QUICK_TEST = true first and report any error.
%
% MAKE_EXTRA_FIGS (Phase 7, default false)
%   true draws the Phase 7 figures after the standard set, under
%   ./sa_appa_2nd_output/figures_phase7/: Part A surrogate fidelity on TP3
%   (ground truth f*(x), theta(x), y*(x); the GP of theta that SA-APPA held
%   after the initial design and at the end of the lower-median M1 run; error
%   maps; held-out RMSE / maximum error / +/- 2 sigma coverage), UL constraint
%   maps for TP3, TP4, TP5, TP7, TP8, held-out fidelity for TP1-TP8 and
%   SMD (3,3); Part B budget composition (UL + LL objective evaluations per
%   run, counted by llEvalCounter() during the sweep) and evaluations to
%   target (1e-2 and 1.0) from the stored trajectories. Tables and captions are
%   written beside the figures (phase7_*.csv, phase7_captions.tsv). Same logic
%   as the Python figures_phase7.py / phase7_data.py, applied to this file's own
%   sweep, so the numbers are MATLAB's, not Python's. M14 (BL-SAEA) is not in
%   this file. With the flag false, the only change to a run is one counter
%   increment per LL objective call, which does not alter any result. The
%   extra figures need extra reference LL solves (held-out sets) and take
%   roughly as long as a few hundred sweep runs.
%
% QUICK_TEST
%   The hardcoded constant QUICK_TEST (below, default false) switches to a
%   smoke test: 3 instances, all 13 methods, 3 runs. The Friedman tests in
%   quick mode use n = 3 blocks and are NOT meaningful statistics; the
%   validation report says so.
%
% RANDOMNESS
%   MATLAB's Mersenne Twister and Python's PCG64 (numpy default_rng) are
%   different generators: the same integer seed does NOT give the same draws.
%   Seeds are derived by the same rule as the Python runner: run_id = 0..29 is
%   the run seed, and every derived stream uses _stable_seed(), i.e. the MD5 of
%   the '|'-joined parts reduced modulo 2^31-1 (stableSeed() below). Each
%   population / nested lower-level solve gets its own RandStream('mt19937ar')
%   (mirroring Python's independent default_rng objects), never a shared global
%   state. Treat a successful translation as STATISTICAL agreement with
%   Python, not numerical identity (see the validation report at the end).
%
% LOWER-LEVEL SOLVER NOTE
%   Python uses scipy.optimize.differential_evolution (best1bin, dithered
%   F in (0.4,1.0), CR = 0.7, popsize 6 * dim, 20 generations, Latin
%   hypercube initialisation, deferred updating, relative-std convergence test
%   tol = 1e-7, quadratic penalty 1e6 on lower-level constraints) followed by
%   a polish=True L-BFGS-B step. llSolve() re-implements the DE loop. Base
%   MATLAB has no bound-constrained local optimiser, so the polish is
%   approximated by a simplified bound-constrained L-BFGS (projected steps,
%   forward-difference gradients, Armijo backtracking; see llPolish()). This is
%   an approximation and one of the places where exact numerical agreement
%   with Python cannot be expected.
%   After the polish, ssSelect() applies the strong Stackelberg selection of
%   strong_stackelberg.py (the main lower-level protocol of the paper): among
%   tied optimal LL replies it returns the one best for the leader, with the
%   same accuracy gate, validation, materiality and tie tests. Its local
%   solves need fmincon (Optimization Toolbox) in MATLAB; without it the step
%   is skipped with a warning, and the run follows the no-selection protocol.
%
% NO COPIED PYTHON VALUES
%   No numeric result in this file is a copied-in Python value. Every printed
%   number and every plotted point comes from MATLAB executing its own
%   benchmark functions, its own 13 methods and its own statistics. The ONLY
%   Python numbers ever read are scenario_tables/scenario_{1..7}.csv, loaded
%   at the very end purely as a static reference for the validation report.
%   The only other numbers hardcoded are the benchmark definitions, the
%   algorithm constants, and the Friedmantest.pdf worked-example inputs used
%   by the formula self-check (as in the Python friedman_test.py).
%
% OUTPUTS (written under ./sa_appa_2nd_output/)
%   scenario_tables/scenario_{1..7}.csv  Best, Mean, Std, CPU per method per
%                                         instance (Python's column convention:
%                                         <M>_best, <M>_mean, <M>_std, <M>_cpu,
%                                         <M>_nfeas); also printed to console.
%   summary.csv                          one row per (method, instance, run).
%   figures/*.eps and *.pdf              opaque fills, no alpha:
%       fig1a-d  solution-vs-method boxplots (TP1-TP9, SMD@(3,3), (10,10),
%                (30,30) as four separate figures)
%       fig2     performance profile       fig2b  data profile (tau 0.1, 0.25)
%       fig3_*   convergence curves        fig4_* rate of convergence
%       fig_cd_scenario{1..7}  critical-difference diagrams
%
% EXPECTED FULL-RUN WALL-CLOCK TIME (estimate; not measured in MATLAB)
%   The Python sweep recorded 17,160 runs (13 methods x 44 instances x 30 runs)
%   with a mean cpu_time_s of about 10.7 s per run (median 8.2 s, maximum 95 s
%   on the 30-dimensional SMD instances), about 51 CPU-hours in total. This
%   script runs serially. The only timing available is from the authoring
%   environment under GNU Octave 8 (a slower interpreter than MATLAB's JIT):
%   2.8, 5.5 and 14.8 s per run on TP1, TP5 and SMD1_p3q3, against Python's
%   0.9, 0.8 and 4.2 s on the same instances, i.e. 3 to 7 times slower. MATLAB
%   should be faster than Octave on these scalar loops, but by how much is
%   unmeasured. Planning range for the full sweep on one core: about 2 days
%   (MATLAB close to Python speed) to about 2 weeks (Octave-like speed).
%   Time QUICK_TEST first (3 instances x 13 methods x 3 runs = 117 runs), divide
%   by 117 and multiply by 17,160 -- but note those 3 instances are cheap ones,
%   so scale that figure up (Python's mean per run over all 44 instances is
%   about 5 times its mean over these 3). The sweep checkpoints to
%   sa_appa_2nd_output/sweep_checkpoint.mat every 100 runs and resumes from it
%   if the file exists (delete the file to start fresh).
%
% TABLE OF CONTENTS (function -- purpose -- line number in this file)
%   Main script body (config, sweep, tables, figures, validation) ----  228
%   --- utilities ---
%   isOctave               -- Octave detection                             332
%   argmaxNP / argminNP    -- numpy-style arg-extrema (first NaN wins)     336
%   normPdf / normCdf      -- standard normal pdf / cdf via erf/erfc       357
%   percentileLocal        -- linear-interpolation percentile              365
%   rankAvg                -- average ranks on ties, 1 = best              375
%   clipRows               -- clip population rows to box bounds           392
%   --- randomness ---
%   stableSeed             -- MD5-based seed rule of Python's runner       400
%   md5Hex                 -- MD5 hex digest (Java / fallback)            415
%   hexToModP              -- big hex integer mod 2^31-1                   438
%   newStream              -- RandStream('mt19937ar') from a seed          448
%   lhsSample              -- Latin hypercube sample in a box              452
%   uniformSample          -- uniform sample in a box                      463
%   --- benchmark suite (44 instances) ---
%   buildSuite             -- assemble TP1-TP9 + SMD1-SMD12 suite          473
%   buildTPInstances       -- TP1-TP9 definitions                          504
%   tp9f                   -- TP9 lower-level objective                    496
%   buildSMDInstances      -- SMD1-SMD12 at three scales                   736
%   makeInstance / addInst -- instance struct constructor and appender     479
%   boxBounds              -- repeat [lo,hi] bounds n times                492
%   rosenbrockZ            -- Rosenbrock helper                            484
%   smdBounds              -- SMD box bounds per base problem              596
%   smdSplit               -- split x,y into xu1,xu2,xl1,xl2               584
%   smdAnn                 -- SMD9 annulus helper                          591
%   smdEval                -- SMD F / f / G / g evaluator                  619
%   evalG / evalLLg        -- constraint evaluators (row-vector output)    760
%   --- surrogates ---
%   sqDist                 -- squared pairwise distances                   772
%   gpFit / gpPredict      -- Gaussian-process surrogate (RBF kernel)      780
%   rbfFit / rbfPredict    -- multiquadric RBF surrogate (M11)             815
%   surrogateFit/Predict   -- dispatch on surrogate kind                   839
%   surrogatePredict       -- surrogate prediction dispatch                847
%   fitSurrogates          -- objective + constraint surrogates            855
%   eiAcq / pofAcq         -- expected improvement / prob. of feasibility  863
%   predictCEI             -- constrained EI over a batch                  877
%   predictPenalizedEI     -- penalised EI over a batch                    892
%   --- lower-level solver ---
%   llSolve                -- DE (scipy-style) nested lower-level solve   1013
%   llObjective            -- penalised lower-level objective              914
%   llPolish               -- bound-constrained L-BFGS polish (L-BFGS-B stand-in)  958
%   llGrad / llTwoLoop     -- finite-difference gradient / L-BFGS recursion  922
%   --- the 13 methods ---
%   appaStep               -- adaptive prey-predator update               1084
%   gaHybridize            -- crossover / mutation replace-worst          1113
%   deStep                 -- DE/rand/1/bin generation (M5, M10)          1136
%   psoStep                -- PSO velocity / position update (M6, M11)    1155
%   trajNew / trajRecord   -- per-run trajectory bookkeeping              1163
%   isFeasibleG            -- feasibility test (all G <= 1e-6)            1178
%   initialDesign          -- N0 initial points with exact LL solves      1182
%   runAblationMethod      -- surrogate-assisted optimiser, M1-M11        1205
%   runNestedMethod        -- non-surrogate nested GA, M12-M13            1350
%   nestedEvaluate         -- exact LL solves for a nested population      1380
%   methodSpecs            -- Table 1 method configurations               1393
%   runMethod              -- dispatch one (method, instance, run)        1414
%   runM1_AppendixD7Diagnostic -- OPTIONAL pf_eta-corrected M1 variant    1427
%   --- sweep and statistics ---
%   runFullSweep           -- methods x instances x runs, checkpointed    1443
%   cellStats              -- Best/Mean/Std/CPU for one method-instance   1525
%   scenarioDefs           -- the 7 scenario definitions                  1541
%   buildScenarioTable     -- Appendix B table for one scenario           1556
%   rankMatrixForScenario  -- per-instance ranks of scenario methods      1572
%   friedmanStat           -- Conover F-approximation Friedman test       1623
%   fUpperP / fCritical    -- F upper-tail p-value / critical value       1589
%   tCritical              -- two-sided t critical value                  1617
%   conoverCD              -- Conover post-hoc CD (rank-sum units)        1643
%   nemenyiCD              -- Nemenyi CD (mean-rank units)                1655
%   srCdfInf               -- studentized range cdf, infinite df          1649
%   bisectRoot             -- bisection root finder                       1594
%   runScenario            -- table + ranks + Friedman for one scenario   1661
%   printScenarioReport    -- console report for one scenario             1679
%   printScenarioTable     -- console Appendix B table                    1702
%   writeScenarioCsv       -- scenario_{n}.csv writer                     1728
%   csvNum                 -- rounded numeric field for CSV output          1720
%   writeSummaryCsv        -- summary.csv writer                          1749
%   validateFriedmanAgainstWorkedExample -- formula self-check            1761
%   --- figures ---
%   methodStyle            -- 13 distinct colour / linestyle / marker     1801
%   newFigure / addLegend / saveFigureBoth -- opaque figure, legend, EPS+PDF export 1822
%   lighten                -- blend a colour toward white (no alpha)      1817
%   bestAndMeans / ffillColumns -- profile helpers (f_L, method means, ffill) 1901
%   drawBoxplot            -- hand-rolled boxplot                         1863
%   makeFigure1            -- four per-group boxplot figures              1917
%   makeFigure2            -- Dolan-More performance profile              1964
%   makeFigure2b           -- More-Wild data profile                      1994
%   makeFigure3One         -- convergence curves for one instance         2055
%   makeFigure4One         -- convergence-rate plot for one instance      2124
%   fitConvergenceOrder    -- log-log OLS convergence order               2108
%   makeCdDiagram          -- critical-difference diagram                 2173
%   --- validation report ---
%   loadPythonScenarioTable -- parse a Python scenario CSV (reference only) 2218
%   validationReport       -- MATLAB-vs-Python agreement table            2272
%   yn                     -- Y/N flag string for the report              2342
%   --- Phase 7 extra figures (MAKE_EXTRA_FIGS) ---
%   llEvalCounter          -- LL objective evaluation counter (B1)
%   ssSelect / ssPick / ssLocalMin / ssLocalLL -- strong Stackelberg selection
%   phase7Options / phase7Style / phase7Label -- settings, styles, labels
%   phase7Ystar / phase7Truth -- exact (TP3, TP8) or reference LL response
%   phase7MedianRun / phase7Rerun -- lower-median run, snapshot re-run and check
%   phase7Metrics          -- RMSE, maximum error, +/- 2 sigma coverage
%   phase7Fig3d / phase7Wire / phase7Seq / phase7Save -- drawing helpers
%   phase7HatchRect / phase7Bars -- hatched grouped bars (opaque)
%   phase7Figures          -- orchestrator
%   phase7PartA_TP3        -- A1, A2, A4 on TP3
%   phase7PartA3 / phase7PartA5 -- constraint maps, held-out fidelity
%   phase7PartA6 -- SMD6: non-unique LL response, diagnostic GP on optimistic targets
%   phase7PartB            -- B1 budget composition, B2 evaluations to target
%   (line numbers above refer to the file before the Phase 7 additions)

%% ===================== MAIN SCRIPT BODY =====================
clear; clc; close all;

QUICK_TEST = false;   % true = 3 instances x 13 methods x 3 runs smoke test
MAKE_EXTRA_FIGS = false;   % true = also draw the Phase 7 figures (surrogate fidelity,
                           % budget composition, evaluations to target); see phase7Figures()

t_script_start = tic;
fprintf('==============================================================\n');
fprintf(' SA_APPA_2nd.m  --  SA-APPA experiment translation (serial)\n');
fprintf('==============================================================\n');
if isOctave()
    error('SA_APPA_2nd:octave', 'This file needs MATLAB R2016b or later (Octave has no RandStream class).');
end
fprintf('QUICK_TEST = %d   MAKE_EXTRA_FIGS = %d\n', QUICK_TEST, MAKE_EXTRA_FIGS);

% ---- fixed experimental configuration (methods.py / ll_solver.py) ----
cfg = struct();
cfg.N0 = 25;            % initial LHS design size
cfg.T_MAX = 35;         % infill generations
cfg.NEST_POP = 6;       % nested methods population
cfg.NEST_GEN = 10;      % nested methods generations (6 x 10 = 60 evaluations)
cfg.N_UL_POP = 15;      % upper-level population for population-based engines
cfg.LL_POPSIZE = 6;     % scipy popsize multiplier (population = 6 * dim)
cfg.LL_MAXITER = 20;    % scipy maxiter

methodIds = arrayfun(@(k) sprintf('M%d', k), 1:13, 'UniformOutput', false);
suite = buildSuite();
allInstanceIds = {suite.id};

if QUICK_TEST
    instanceIds = {'TP1', 'TP5', 'SMD1_p3q3'};
    nRuns = 3;
else
    instanceIds = allInstanceIds;
    nRuns = 30;
end
fprintf('Methods: %d   Instances: %d   Runs per cell: %d   Total runs: %d\n', ...
    numel(methodIds), numel(instanceIds), nRuns, numel(methodIds) * numel(instanceIds) * nRuns);

outDir = fullfile('.', 'sa_appa_2nd_output');
tabDir = fullfile(outDir, 'scenario_tables');
figDir = fullfile(outDir, 'figures');
if ~exist(tabDir, 'dir'), mkdir(tabDir); end
if ~exist(figDir, 'dir'), mkdir(figDir); end

% ---- sanity checks before the long run ----
fprintf('\n[1/6] Suite sanity check (44 instances evaluate to finite values) ...\n');
assert(numel(suite) == 44, 'expected 44 reconciled instances, got %d', numel(suite));
sanityStream = newStream(0);
for k = 1:numel(suite)
    inst = suite(k);
    xs = inst.xb(:, 1)' + rand(sanityStream, 1, inst.n_ul) .* (inst.xb(:, 2) - inst.xb(:, 1))';
    ys = inst.yb(:, 1)' + rand(sanityStream, 1, inst.n_ll) .* (inst.yb(:, 2) - inst.yb(:, 1))';
    vals = [inst.F(xs, ys), inst.f(xs, ys), evalG(inst, xs, ys), evalLLg(inst, xs, ys)];
    assert(all(isfinite(vals)), 'instance %s gave a non-finite value at a random box point', inst.id);
end
fprintf('    OK.\n');
fprintf('[2/6] Friedman formula self-check against Friedmantest.pdf worked example ...\n');
validateFriedmanAgainstWorkedExample();

% ---- the experiment ----
fprintf('\n[3/6] Running the sweep (this is the long step) ...\n');
R = runFullSweep(suite, methodIds, instanceIds, nRuns, cfg);
fprintf('Sweep finished. Elapsed since start: %.1f min\n', toc(t_script_start) / 60);
writeSummaryCsv(R, fullfile(outDir, 'summary.csv'));

% ---- scenario tables and statistics ----
fprintf('\n[4/6] Scenario tables and Friedman statistics ...\n');
scen = scenarioDefs();
S = cell(1, numel(scen));
for sIdx = 1:numel(scen)
    S{sIdx} = runScenario(R, scen(sIdx));
    printScenarioReport(S{sIdx});
    printScenarioTable(S{sIdx});
    writeScenarioCsv(S{sIdx}, fullfile(tabDir, sprintf('scenario_%d.csv', scen(sIdx).num)));
end

% ---- figures ----
fprintf('\n[5/6] Figures (EPS + PDF, opaque, no alpha) ...\n');
makeFigure1(R, figDir);
makeFigure2(R, figDir);
makeFigure2b(R, suite, figDir, 0.1);
makeFigure2b(R, suite, figDir, 0.25);
% Figures 3 and 4: the manuscript's own subset (TP2 and SMD4 at all three
% scales). Set fig34InstanceIds = instanceIds for the full 44-instance set.
fig34InstanceIds = intersect({'TP2', 'SMD4_p3q3', 'SMD4_p10q10', 'SMD4_p30q30'}, instanceIds, 'stable');
if QUICK_TEST, fig34InstanceIds = instanceIds(1:2); end
for k = 1:numel(fig34InstanceIds)
    makeFigure3One(R, fig34InstanceIds{k}, figDir);
    makeFigure4One(R, fig34InstanceIds{k}, figDir);
end
for sIdx = 1:numel(scen)
    makeCdDiagram(S{sIdx}, figDir);
end
fprintf('    figures written to %s\n', figDir);
if MAKE_EXTRA_FIGS
    fprintf('\n[5b/6] Phase 7 figures (surrogate fidelity, budget composition, evaluations to target) ...\n');
    phase7Figures(R, suite, cfg, fullfile(outDir, 'figures_phase7'), QUICK_TEST);
end

% ---- self-validation report ----
fprintf('\n[6/6] Self-validation report: MATLAB statistics vs Python reference tables ...\n');
validationReport(S, scen, instanceIds, QUICK_TEST);

fprintf('\nTotal elapsed: %.1f min. Outputs in %s\n', toc(t_script_start) / 60, outDir);

%% ===================== UTILITIES =====================
% Local functions begin here. Everything above this line is the script body.

function tf = isOctave()
    tf = exist('OCTAVE_VERSION', 'builtin') ~= 0;
end

function idx = argmaxNP(v)
    % numpy.argmax: first NaN wins if any NaN is present, else first maximum.
    v = v(:);
    nanIdx = find(isnan(v), 1);
    if ~isempty(nanIdx)
        idx = nanIdx;
    else
        [~, idx] = max(v);
    end
end

function idx = argminNP(v)
    v = v(:);
    nanIdx = find(isnan(v), 1);
    if ~isempty(nanIdx)
        idx = nanIdx;
    else
        [~, idx] = min(v);
    end
end

function p = normPdf(x)
    p = exp(-0.5 * x .^ 2) / sqrt(2 * pi);
end

function c = normCdf(x)
    c = 0.5 * erfc(-x / sqrt(2));
end

function q = percentileLocal(v, p)
    % Linear interpolation between order statistics (numpy default / R-7).
    v = sort(v(:));
    n = numel(v);
    if n == 0, q = NaN; return; end
    pos = 1 + (n - 1) * p / 100;
    lo = floor(pos); hi = ceil(pos);
    q = v(lo) + (pos - lo) * (v(hi) - v(lo));
end

function r = rankAvg(v)
    % Average ranks on ties, 1 = smallest (best under minimisation).
    v = v(:)';
    n = numel(v);
    [sv, order] = sort(v);
    r = zeros(1, n);
    i = 1;
    while i <= n
        j = i;
        while j < n && sv(j + 1) == sv(i)
            j = j + 1;
        end
        r(order(i:j)) = (i + j) / 2;
        i = j + 1;
    end
end

function X = clipRows(X, bounds)
    for c = 1:size(X, 2)
        X(:, c) = min(max(X(:, c), bounds(c, 1)), bounds(c, 2));
    end
end

%% ===================== RANDOMNESS =====================

function seed = stableSeed(varargin)
    % Mirrors methods.py _stable_seed(*parts): md5 of "|".join(str(p)) mod 2^31-1.
    parts = cell(1, numel(varargin));
    for k = 1:numel(varargin)
        v = varargin{k};
        if ischar(v)
            parts{k} = v;
        else
            parts{k} = sprintf('%d', v);
        end
    end
    key = strjoin(parts, '|');
    seed = hexToModP(md5Hex(key), 2 ^ 31 - 1);
end

function h = md5Hex(str)
    % MD5 hex digest via Java MessageDigest. Last-resort fallback (no JVM): a
    % deterministic polynomial hash -- NOT MD5, so derived seeds then differ
    % from Python's, which only affects seed bookkeeping.
    h = '';
    try
        md = java.security.MessageDigest.getInstance('MD5');
        md.update(uint8(str));
        d = typecast(md.digest(), 'uint8');
        h = lower(reshape(dec2hex(d, 2)', 1, []));
    catch
        h = '';
    end
    if isempty(h)
        warning('md5Hex:fallback', 'No MD5 available; using a polynomial hash for seed derivation.');
        acc = 7;
        for k = 1:numel(str)
            acc = mod(acc * 31 + double(str(k)), 2 ^ 31 - 1);
        end
        h = dec2hex(acc);
    end
end

function r = hexToModP(h, p)
    % Big hexadecimal integer reduced modulo p, digit by digit (exact in doubles
    % because acc < 2^31 and acc * 16 < 2^35).
    acc = 0;
    for k = 1:numel(h)
        acc = mod(acc * 16 + hex2dec(h(k)), p);
    end
    r = acc;
end

function s = newStream(seed)
    s = RandStream('mt19937ar', 'Seed', seed);
end

function X = lhsSample(stream, bounds, n)
    % Latin hypercube sample of n points in the box [bounds(:,1), bounds(:,2)].
    d = size(bounds, 1);
    U = zeros(n, d);
    for c = 1:d
        perm = randperm(stream, n)';
        U(:, c) = ((perm - 1) + rand(stream, n, 1)) / n;
    end
    X = bounds(:, 1)' + U .* (bounds(:, 2) - bounds(:, 1))';
end

function X = uniformSample(stream, bounds, n)
    d = size(bounds, 1);
    X = bounds(:, 1)' + rand(stream, n, d) .* (bounds(:, 2) - bounds(:, 1))';
end

%% ===================== BENCHMARK SUITE (44 INSTANCES) =====================
% Every instance exposes F, f (scalar upper / lower objectives) and G, g
% (constraint vectors, feasible iff all <= 0), as handles taking row vectors
% x, y. Formulas are transcribed from benchmark.py.

function suite = buildSuite()
    tp = buildTPInstances();
    smd = buildSMDInstances();
    suite = [tp, smd];
end

function inst = makeInstance(id, family, n_ul, n_ll, xb, yb, F, f, G, g, scale, group)
    inst = struct('id', id, 'family', family, 'n_ul', n_ul, 'n_ll', n_ll, ...
        'xb', xb, 'yb', yb, 'F', F, 'f', f, 'G', G, 'g', g, 'scale', scale, 'group', group);
end

function z = rosenbrockZ(v)
    if numel(v) < 2
        z = 0;
    else
        z = sum((v(2:end) - v(1:end-1) .^ 2) .^ 2 + (v(1:end-1) - 1) .^ 2);
    end
end

function b = boxBounds(lo, hi, n)
    b = repmat([lo, hi], n, 1);
end

function f = tp9f(x, y)
    EPS = 1e-9;
    yy = y;
    yy(abs(y) < EPS) = EPS;
    idx = 1:numel(x);
    f = exp(1 + (1 / 4000) * sum(x .^ 2 .* y .^ 2) - prod(cos(x ./ (yy .* sqrt(idx)))));
end

function insts = buildTPInstances()
    EPS = 1e-9;
    insts = struct([]);

    % TP1
    insts = addInst(insts, makeInstance('TP1', 'TP', 2, 1, boxBounds(-2, 2, 2), boxBounds(-2, 2, 1), ...
        @(x, y) x(1) ^ 2 + x(2) ^ 2, ...
        @(x, y) 2 * x(1) + y(1), ...
        @(x, y) [], ...
        @(x, y) [x(1) ^ 2 + y(1) ^ 2 - 4, x(1) - y(1)], '', 'TP1-TP9'));

    % TP2
    insts = addInst(insts, makeInstance('TP2', 'TP', 2, 3, boxBounds(0, 10, 2), boxBounds(0, 10, 3), ...
        @(x, y) (1 + x(1) - x(2) + 2 * y(2)) * (8 - x(1) - 2 * y(1) + y(2) + 5 * y(3)), ...
        @(x, y) 2 * y(1) - y(2) + y(3), ...
        @(x, y) [], ...
        @(x, y) [-y(1) + y(2) + y(3) - 1, ...
                 2 * x(1) - y(1) + 2 * y(2) - 0.5 * y(3) - 1, ...
                 -(2 * x(2) + 2 * y(1) - y(2) - 0.5 * y(3) - 1)], '', 'TP1-TP9'));

    % TP3 (box [0,15]^2 already includes the audited x >= 0 lower bound)
    insts = addInst(insts, makeInstance('TP3', 'TP', 2, 2, boxBounds(0, 15, 2), boxBounds(0, 10, 2), ...
        @(x, y) (x(1) - 30) ^ 2 + (x(2) - 20) ^ 2 - 20 * y(1) + 20 * y(2), ...
        @(x, y) (x(1) - y(1)) ^ 2 + (x(2) - y(2)) ^ 2, ...
        @(x, y) [-(x(1) + 2 * x(2) - 30), x(1) + x(2) - 25], ...
        @(x, y) [], '', 'TP1-TP9'));

    % TP4
    insts = addInst(insts, makeInstance('TP4', 'TP', 2, 2, boxBounds(0, 10, 2), boxBounds(0, 10, 2), ...
        @(x, y) -x(1) ^ 2 - 3 * x(2) ^ 2 - 4 * y(1) + y(2) ^ 2, ...
        @(x, y) 2 * x(1) ^ 2 + y(1) ^ 2 - 5 * y(2), ...
        @(x, y) [x(1) ^ 2 + 2 * x(2) - 4], ...
        @(x, y) [-(x(1) ^ 2 - 2 * x(1) + x(2) ^ 2 - 2 * y(1) + y(2) + 3), ...
                 4 - (x(2) + 3 * y(1) - 4 * y(2))], '', 'TP1-TP9'));

    % TP5
    insts = addInst(insts, makeInstance('TP5', 'TP', 2, 2, boxBounds(0, 10, 2), boxBounds(0, 10, 2), ...
        @(x, y) -((x(1) + y(1)) * (x(2) + y(2))) / (1 + x(1) * y(1) + x(2) * y(2) + EPS), ...
        @(x, y) -((x(1) + y(1)) * (x(2) + y(2))) / (1 + x(1) * y(1) + x(2) * y(2) + EPS), ...
        @(x, y) [x(1) - x(2), x(1) ^ 2 + x(2) ^ 2 - 100], ...
        @(x, y) [-y(1), y(1) - x(1), -y(2), y(2) - x(2)], '', 'TP1-TP9'));

    % TP6: n_ul = 1 (x2), n_ll = 2 (x1, x3); f does not depend on x2 by the paper's formula
    insts = addInst(insts, makeInstance('TP6', 'TP', 1, 2, boxBounds(log(0.1), log(10), 1), boxBounds(log(0.1), log(10), 2), ...
        @(x, y) exp(x(1)), ...
        @(x, y) exp(-y(1)) + 0.5864 * exp(0.6 * y(1)), ...
        @(x, y) [], ...
        @(x, y) [0.0332 * exp(y(2)) + 0.1 * exp(y(1)) - 1, ...
                 4 * exp(x(1) - y(2)) + 2 * exp(-0.71 * x(1) - y(2)) + 0.0332 * exp(-1.3 * x(1)) - 1], '', 'TP1-TP9'));

    % TP7 (reformulated per Task 2: arg max negated to arg min)
    insts = addInst(insts, makeInstance('TP7', 'TP', 2, 2, boxBounds(0, 1, 2), boxBounds(0, 1, 2), ...
        @(x, y) (x(1) - 1) ^ 2 + (x(2) - 1) ^ 2 + (y(1) - 1) ^ 2, ...
        @(x, y) -(y(1) - x(1) ^ 2) ^ 2 - (y(2) - x(2) ^ 2) ^ 2, ...
        @(x, y) [x(1) ^ 2 + x(2) ^ 2 - 1], ...
        @(x, y) [], '', 'TP1-TP9'));

    % TP8
    insts = addInst(insts, makeInstance('TP8', 'TP', 2, 2, boxBounds(0, 50, 2), boxBounds(-10, 20, 2), ...
        @(x, y) abs(2 * x(1) + 2 * x(2) - 3 * y(1) - 3 * y(2) - 60), ...
        @(x, y) (y(1) - x(1) + 20) ^ 2 + (y(2) - x(2) + 20) ^ 2, ...
        @(x, y) [x(1) + x(2) + y(1) - 2 * y(2) - 40], ...
        @(x, y) [10 - x(1) + 2 * y(1), 10 - x(2) + 2 * y(2)], '', 'TP1-TP9'));

    % TP9
    insts = addInst(insts, makeInstance('TP9', 'TP', 10, 10, boxBounds(-5, 5, 10), boxBounds(-pi, pi, 10), ...
        @(x, y) sum(abs(x - 1) + abs(y)), ...
        @(x, y) tp9f(x, y), ...
        @(x, y) [], ...
        @(x, y) [], '', 'TP1-TP9'));
end

function arr = addInst(arr, inst)
    if isempty(arr)
        arr = inst;
    else
        arr(end + 1) = inst;
    end
end

function [xu1, xu2, xl1, xl2] = smdSplit(x, y, p, r, q, s)
    xu1 = x(1:p);
    xu2 = x(p + 1:p + r);
    xl1 = y(1:q + s);
    xl2 = y(q + s + 1:q + s + r);
end

function a = smdAnn(u1, u2, aa, bb)
    s_ = (sum(u1 .^ 2) + sum(u2 .^ 2)) / aa;
    a = s_ - floor(s_ + 1 / (2 * bb));
end

function [xb, yb] = smdBounds(baseIdx, p, q, r, s)
    xu1b = boxBounds(-5, 10, p);
    xu2b = boxBounds(-5, 10, r);
    xl1b = boxBounds(-5, 10, q + s);
    xl2b = boxBounds(-pi / 2 + 1e-6, pi / 2 - 1e-6, r);
    switch baseIdx
        case 2
            xu2b = boxBounds(-5, 1, r);  xl2b = boxBounds(1e-6, exp(1), r);
        case 4
            xu2b = boxBounds(-1, 1, r);  xl2b = boxBounds(0, exp(1), r);
        case 7
            xu2b = boxBounds(-5, 1, r);  xl2b = boxBounds(1e-6, exp(1), r);
        case 9
            xu2b = boxBounds(-5, 1, r);  xl2b = boxBounds(-1 + 1e-6, -1 + exp(1), r);
        case 11
            xu2b = boxBounds(-1, 1, r);  xl2b = boxBounds(1 / exp(1), exp(1), r);
        case 12
            xu2b = boxBounds(-14.10, 14.10, r);  xl2b = boxBounds(-1.5 + 1e-6, 1.5 - 1e-6, r);
    end
    xb = [xu1b; xu2b];
    yb = [xl1b; xl2b];
end

function out = smdEval(baseIdx, which, x, y, p, q, r, s)
    % which: 1 = F, 2 = f, 3 = G, 4 = g. baseIdx = 1..12 (SMD1..SMD12).
    [xu1, xu2, xl1, xl2] = smdSplit(x, y, p, r, q, s);
    out = [];
    switch baseIdx
        case 1
            if which == 1
                out = sum(xu1 .^ 2) + sum(xl1 .^ 2) + sum(xu2 .^ 2) + sum((xu2 - tan(xl2)) .^ 2);
            elseif which == 2
                out = sum(xu1 .^ 2) + sum(xl1 .^ 2) + sum((xu2 - tan(xl2)) .^ 2);
            end
        case 2
            if which == 1
                out = sum(xu1 .^ 2) - sum(xl1 .^ 2) + sum(xu2 .^ 2) - sum((xu2 - log(xl2)) .^ 2);
            elseif which == 2
                out = sum(xu1 .^ 2) + sum(xl1 .^ 2) + sum((xu2 - log(xl2)) .^ 2);
            end
        case 3
            if which == 1
                out = sum(xu1 .^ 2) + sum(xl1 .^ 2) + sum(xu2 .^ 2) + sum((xu2 .^ 2 - tan(xl2)) .^ 2);
            elseif which == 2
                rastrigin = q + sum(xl1 .^ 2 - cos(2 * pi * xl1));
                out = sum(xu1 .^ 2) + rastrigin + sum((xu2 .^ 2 - tan(xl2)) .^ 2);
            end
        case 4
            if which == 1
                out = sum(xu1 .^ 2) - sum(xl1 .^ 2) + sum(xu2 .^ 2) - sum((abs(xu2) - log(1 + xl2)) .^ 2);
            elseif which == 2
                rastrigin = q + sum(xl1 .^ 2 - cos(2 * pi * xl1));
                out = sum(xu1 .^ 2) + rastrigin + sum((abs(xu2) - log(1 + xl2)) .^ 2);
            end
        case 5
            if which == 1
                out = sum(xu1 .^ 2) - rosenbrockZ(xl1) + sum(xu2 .^ 2) - sum((abs(xu2) - xl2 .^ 2) .^ 2);
            elseif which == 2
                out = sum(xu1 .^ 2) + rosenbrockZ(xl1) + sum((abs(xu2) - xl2 .^ 2) .^ 2);
            end
        case 6
            if which == 1
                out = sum(xu1 .^ 2) - sum(xl1(1:q) .^ 2) + sum(xl1(q + 1:q + s) .^ 2) ...
                    + sum(xu2 .^ 2) - sum((xu2 - xl2) .^ 2);
            elseif which == 2
                tail = xl1(q + 1:q + s);
                if s > 1
                    tailTerm = sum((tail(2:end) - tail(1:end-1)) .^ 2);
                else
                    tailTerm = 0;
                end
                out = sum(xu1 .^ 2) + sum(xl1(1:q) .^ 2) + tailTerm + sum((xu2 - xl2) .^ 2);
            end
        case 7
            if which == 1
                idx = 1:p;
                griewank = 1 + (1 / 400) * sum(xu1 .^ 2) - prod(cos(xu1 ./ sqrt(idx)));
                out = griewank - sum(xl1 .^ 2) + sum(xu2 .^ 2) - sum((xu2 - log(xl2)) .^ 2);
            elseif which == 2
                out = sum(xu1 .^ 3) + sum(xl1 .^ 2) + sum((xu2 - log(xl2)) .^ 2);
            end
        case 8
            if which == 1
                ackley = 20 + exp(1) - 20 * exp(-0.2 * sqrt(mean(xu1 .^ 2))) - exp(mean(cos(2 * pi * xu1)));
                out = ackley - rosenbrockZ(xl1) + sum(xu2 .^ 2) - sum((xu2 - xl2 .^ 3) .^ 2);
            elseif which == 2
                out = sum(abs(xu1)) + rosenbrockZ(xl1) + sum((xu2 - xl2 .^ 3) .^ 2);
            end
        case 9
            if which == 1
                out = sum(xu1 .^ 2) - sum(xl1 .^ 2) + sum(xu2 .^ 2) - sum((xu2 - log(1 + xl2)) .^ 2);
            elseif which == 2
                out = sum(xu1 .^ 2) + sum(xl1 .^ 2) + sum((xu2 - log(1 + xl2)) .^ 2);
            elseif which == 3
                out = -smdAnn(xu1, xu2, 1.0, 1.0);
            else
                out = -smdAnn(xl1, xl2, 1.0, 1.0);
            end
        case 10
            if which == 1
                out = sum((xu1 - 2) .^ 2) + sum(xl1 .^ 2) + sum((xu2 - 2) .^ 2) - sum((xu2 - tan(xl2)) .^ 2);
            elseif which == 2
                out = sum(xu1 .^ 2) + sum((xl1 - 2) .^ 2) + sum((xu2 - tan(xl2)) .^ 2);
            elseif which == 3
                su1 = sum(xu1 .^ 3); su2 = sum(xu2 .^ 3);
                out = [-(xu1 - (su1 - xu1 .^ 3) - su2), -(xu2 - (su2 - xu2 .^ 3) - su1)];
            else
                sl1 = sum(xl1 .^ 3);
                out = -(xl1 - (sl1 - xl1 .^ 3));
            end
        case 11
            if which == 1
                out = sum(xu1 .^ 2) - sum(xl1 .^ 2) + sum(xu2 .^ 2) - sum((xu2 - log(xl2)) .^ 2);
            elseif which == 2
                out = sum(xu1 .^ 2) + sum(xl1 .^ 2) + sum((xu2 - log(xl2)) .^ 2);
            elseif which == 3
                arg = max(1.0 / r + log(xl2), 0.0);   % domain safety guard (as in benchmark.py)
                out = sqrt(arg) - xu2;
            else
                out = 1.0 - sum((xu2 - log(xl2)) .^ 2);
            end
        case 12
            if which == 1
                out = sum((xu1 - 2) .^ 2) + sum(xl1 .^ 2) + sum((xu2 - 2) .^ 2) ...
                    + sum(tan(abs(xl2))) - sum((xu2 - tan(xl2)) .^ 2);
            elseif which == 2
                out = sum(xu1 .^ 2) + sum((xl1 - 2) .^ 2) + sum((xu2 - tan(xl2)) .^ 2);
            elseif which == 3
                su1 = sum(xu1 .^ 3); su2 = sum(xu2 .^ 3);
                out = [tan(xl2) - xu2, -(xu1 - (su1 - xu1 .^ 3) - su2), -(xu2 - (su2 - xu2 .^ 3) - su1)];
            else
                sl1 = sum(xl1 .^ 3);
                out = [1.0 - sum((xu2 - tan(xl2)) .^ 2), -(xl1 - (sl1 - xl1 .^ 3))];
            end
    end
    if which >= 3 && isempty(out)
        out = [];
    end
end

function insts = buildSMDInstances()
    scales = [3 3; 10 10; 30 30];
    r = 2;
    insts = struct([]);
    for i = 1:12
        if i == 6, s = 2; else, s = 0; end
        for sc = 1:size(scales, 1)
            p = scales(sc, 1); q = scales(sc, 2);
            if i == 10 && p == 30 && q == 30
                continue;   % excluded (Task 2 / Appendix C)
            end
            [xb, yb] = smdBounds(i, p, q, r, s);
            bi = i; pp = p; qq = q; rr = r; ss = s;
            Fh = @(x, y) smdEval(bi, 1, x, y, pp, qq, rr, ss);
            fh = @(x, y) smdEval(bi, 2, x, y, pp, qq, rr, ss);
            Gh = @(x, y) smdEval(bi, 3, x, y, pp, qq, rr, ss);
            gh = @(x, y) smdEval(bi, 4, x, y, pp, qq, rr, ss);
            id = sprintf('SMD%d_p%dq%d', i, p, q);
            insts = addInst(insts, makeInstance(id, 'SMD', p + r, (q + s) + r, xb, yb, ...
                Fh, fh, Gh, gh, sprintf('(%d,%d)', p, q), sprintf('SMD @ (%d,%d)', p, q)));
        end
    end
end

function Gv = evalG(inst, x, y)
    Gv = inst.G(x, y);
    Gv = Gv(:)';
end

function gv = evalLLg(inst, x, y)
    gv = inst.g(x, y);
    gv = gv(:)';
end

%% ===================== SURROGATES (gp.py) AND ACQUISITION (methods.py) =====================

function D2 = sqDist(A, B)
    % Squared Euclidean distances between the rows of A (n x d) and B (m x d).
    D2 = zeros(size(A, 1), size(B, 1));
    for c = 1:size(A, 2)
        D2 = D2 + (A(:, c) - B(:, c)') .^ 2;
    end
end

function model = gpFit(X, y)
    % GP with an RBF kernel; lengthscale by the median heuristic on the data
    % passed in (re-estimated at every fit, exactly as _fit_surrogates() builds
    % a fresh GPSurrogate each generation), signal variance 1, noise 1e-6.
    JITTER = 1e-8;
    y = y(:);
    n = size(X, 1);
    if n > 1
        d = sqrt(sqDist(X, X));
        mask = triu(true(n), 1);
        med = median(d(mask));
        lengthscale = max(med, 1e-3);
    else
        lengthscale = 1.0;
    end
    signalVar = 1.0;
    noiseVar = 1e-6;
    ymean = mean(y);
    yc = y - ymean;
    K = signalVar * exp(-0.5 * sqDist(X, X) / (lengthscale ^ 2)) + (noiseVar + JITTER) * eye(n);
    L = chol(K, 'lower');
    alpha = L' \ (L \ yc);
    model = struct('kind', 'gp', 'X', X, 'lengthscale', lengthscale, 'signalVar', signalVar, ...
        'ymean', ymean, 'L', L, 'alpha', alpha);
end

function [mu, sd] = gpPredict(model, Xs)
    Ks = model.signalVar * exp(-0.5 * sqDist(Xs, model.X) / (model.lengthscale ^ 2));
    mu = model.ymean + Ks * model.alpha;
    v = model.L \ Ks';
    vr = model.signalVar - sum(v .^ 2, 1)';
    vr = max(vr, 1e-12);
    sd = sqrt(vr);
end

function model = rbfFit(X, y)
    % Multiquadric RBF interpolant (M11, SPSO-RBF); no native predictive variance.
    JITTER = 1e-8;
    y = y(:);
    n = size(X, 1);
    d = sqrt(sqDist(X, X));
    if n > 1
        mask = triu(true(n), 1);
        eps0 = max(median(d(mask)), 1e-3);
    else
        eps0 = 1.0;
    end
    Phi = sqrt(1.0 + sqDist(X, X) / eps0 ^ 2) + JITTER * eye(n);
    w = Phi \ y;
    model = struct('kind', 'rbf', 'X', X, 'epsilon', eps0, 'w', w);
end

function [mu, sd] = rbfPredict(model, Xs)
    d2 = sqDist(Xs, model.X);
    Phi = sqrt(1.0 + d2 / model.epsilon ^ 2);
    mu = Phi * model.w;
    sd = sqrt(min(d2, [], 2));   % distance-to-nearest-training-point exploration proxy
end

function model = surrogateFit(kind, X, y)
    if strcmp(kind, 'rbf')
        model = rbfFit(X, y);
    else
        model = gpFit(X, y);
    end
end

function [mu, sd] = surrogatePredict(model, Xs)
    if strcmp(model.kind, 'rbf')
        [mu, sd] = rbfPredict(model, Xs);
    else
        [mu, sd] = gpPredict(model, Xs);
    end
end

function [theta, phi] = fitSurrogates(X, Ftr, Gtr, mUl, kind)
    theta = surrogateFit(kind, X, Ftr);
    phi = cell(1, mUl);
    for j = 1:mUl
        phi{j} = surrogateFit(kind, X, Gtr(:, j));
    end
end

function ei = eiAcq(mu, sigma, fBest)
    sigma = max(sigma, 1e-12);
    z = (fBest - mu) ./ sigma;
    ei = (fBest - mu) .* normCdf(z) + sigma .* normPdf(z);
end

function p = pofAcq(muList, sigmaList)
    p = 1.0;
    for k = 1:numel(muList)
        sg = max(sigmaList{k}, 1e-12);
        p = p .* normCdf(-muList{k} ./ sg);
    end
end

function [acq, mu, sigma] = predictCEI(cands, theta, phi, fBest)
    [mu, sigma] = surrogatePredict(theta, cands);
    ei = eiAcq(mu, sigma, fBest);
    if ~isempty(phi)
        muL = cell(1, numel(phi)); sgL = cell(1, numel(phi));
        for j = 1:numel(phi)
            [muL{j}, sgL{j}] = surrogatePredict(phi{j}, cands);
        end
        pof = pofAcq(muL, sgL);
    else
        pof = ones(size(mu));
    end
    acq = ei .* pof;
end

function [acq, mu, sigma] = predictPenalizedEI(cands, theta, phi, fBest, t, tMax)
    [mu, sigma] = surrogatePredict(theta, cands);
    ei = eiAcq(mu, sigma, fBest);
    if ~isempty(phi)
        pen = zeros(size(cands, 1), 1);
        for j = 1:numel(phi)
            [muG, ~] = surrogatePredict(phi{j}, cands);
            pen = pen + max(muG, 0.0) .^ 2;
        end
        alphaT = (t / max(tMax, 1)) ^ 2;
        acq = ei - alphaT * pen;
    else
        acq = ei;
    end
end

%% ===================== LOWER-LEVEL SOLVER (ll_solver.py) =====================
% scipy.optimize.differential_evolution re-implemented: best1bin, dithered F in
% (0.4, 1.0) drawn once per generation, CR = 0.7, population = max(5, popsize*dim),
% Latin hypercube initialisation, deferred updating, out-of-box trial components
% re-drawn uniformly, relative-std convergence test, quadratic penalty 1e6.

function val = llObjective(inst, x, y)
    llEvalCounter(1);   % counts calls to the LL objective f (Phase 7, B1); does not change val
    val = inst.f(x, y);
    g = evalLLg(inst, x, y);
    if ~isempty(g)
        val = val + 1e6 * sum(max(g, 0.0) .^ 2);
    end
end

function g = llGrad(fun, y, f0, lo, hi)
    % Forward-difference gradient (relative step sqrt(eps), as scipy's 2-point
    % scheme); a backward step is used where the forward step leaves the box.
    n = numel(y);
    g = zeros(n, 1);
    for k = 1:n
        h = 1.4901161193847656e-08 * max(1, abs(y(k)));
        if y(k) + h > hi(k), h = -h; end
        yk = y; yk(k) = y(k) + h;
        g(k) = (fun(yk) - f0) / h;
    end
end

function r = llTwoLoop(q, S, Yh, free)
    % L-BFGS two-loop recursion restricted to the free variables.
    m = size(S, 2);
    q(~free) = 0;
    a = zeros(1, m); rho = zeros(1, m);
    for i = m:-1:1
        rho(i) = 1 / (Yh(:, i)' * S(:, i));
        a(i) = rho(i) * (S(:, i)' * q);
        q = q - a(i) * Yh(:, i);
    end
    if m > 0
        gam = (S(:, m)' * Yh(:, m)) / (Yh(:, m)' * Yh(:, m));
    else
        gam = 1;
    end
    r = gam * q;
    for i = 1:m
        b = rho(i) * (Yh(:, i)' * r);
        r = r + S(:, i) * (a(i) - b);
    end
    r(~free) = 0;
end

function [yBest, fBest] = llPolish(inst, x, y0, lo, hi, f0)
    % Stand-in for scipy's polish=True L-BFGS-B step: a simplified bound-
    % constrained L-BFGS (projected steps, free-variable two-loop recursion,
    % Armijo backtracking, forward-difference gradients; stops on projected-
    % gradient norm < 1e-5 or relative decrease <= 2.2e-9, as L-BFGS-B's
    % defaults). Accepted only if it improves on the DE result (as scipy does).
    lo = lo(:); hi = hi(:);
    fun = @(v) llObjective(inst, x, v(:)');
    maxIt = 100; mem = 10; ftol = 2.220446049250313e-09; gtol = 1e-5;
    y = y0(:); f = f0;
    g = llGrad(fun, y, f, lo, hi);
    n = numel(y);
    S = zeros(n, 0); Yh = zeros(n, 0);
    for it = 1:maxIt
        atLo = (y <= lo) & (g > 0);
        atHi = (y >= hi) & (g < 0);
        free = ~(atLo | atHi);
        pg = g; pg(~free) = 0;
        if max(abs(pg)) < gtol, break; end
        d = -llTwoLoop(pg, S, Yh, free);
        if d' * pg >= 0
            d = -pg; S = zeros(n, 0); Yh = zeros(n, 0);
        end
        if it == 1 || isempty(S)
            t = min(1, 1 / max(1e-12, norm(pg)));
        else
            t = 1;
        end
        accepted = false;
        for ls = 1:30
            yn = min(max(y + t * d, lo), hi);
            fn = fun(yn);
            if isfinite(fn) && fn <= f + 1e-4 * (g' * (yn - y))
                accepted = true; break;
            end
            t = t / 2;
        end
        if ~accepted, break; end
        gn = llGrad(fun, yn, fn, lo, hi);
        s = yn - y; yv = gn - g;
        if s' * yv > 1e-10 * norm(s) * norm(yv)
            S = [S, s]; Yh = [Yh, yv]; %#ok<AGROW>
            if size(S, 2) > mem, S(:, 1) = []; Yh(:, 1) = []; end
        end
        converged = (f - fn) <= ftol * max([abs(f), abs(fn), 1]);
        y = yn; f = fn; g = gn;
        if converged, break; end
    end
    if f < f0
        yBest = y(:)'; fBest = f;
    else
        yBest = y0; fBest = f0;
    end
end

function [yStar, Fv, Gv, feasLL] = llSolve(inst, x, seed, popsize, maxiter)
    tolConv = 1e-7;
    CR = 0.7;
    lo = inst.yb(:, 1)';
    hi = inst.yb(:, 2)';
    span = hi - lo;
    n = size(inst.yb, 1);
    NP = max(5, popsize * n);
    stream = newStream(seed);

    U = zeros(NP, n);
    for c = 1:n
        U(:, c) = ((randperm(stream, NP)' - 1) + rand(stream, NP, 1)) / NP;
    end
    energies = zeros(NP, 1);
    for i = 1:NP
        energies(i) = llObjective(inst, x, lo + U(i, :) .* span);
    end

    for gen = 1:maxiter
        [~, bi] = min(energies);
        best = U(bi, :);
        scale = 0.4 + 0.6 * rand(stream);
        trials = zeros(NP, n);
        for i = 1:NP
            others = [1:i-1, i+1:NP];
            pick = others(randperm(stream, NP - 1, 2));
            bprime = best + scale * (U(pick(1), :) - U(pick(2), :));
            fillPt = randi(stream, n);
            cross = rand(stream, 1, n) < CR;
            cross(fillPt) = true;
            trial = U(i, :);
            trial(cross) = bprime(cross);
            bad = trial < 0 | trial > 1;
            if any(bad)
                trial(bad) = rand(stream, 1, nnz(bad));
            end
            trials(i, :) = trial;
        end
        for i = 1:NP
            e = llObjective(inst, x, lo + trials(i, :) .* span);
            if e <= energies(i)
                U(i, :) = trials(i, :);
                energies(i) = e;
            end
        end
        if std(energies, 1) <= tolConv * abs(mean(energies))
            break;
        end
    end

    [fDE, bi] = min(energies);
    yDE = lo + U(bi, :) .* span;
    [yStar, ~] = llPolish(inst, x, yDE, lo, hi, fDE);
    yStar = min(max(yStar, lo), hi);
    yStar = ssSelect(inst, x, yStar);   % strong Stackelberg selection among tied LL optima

    gv = evalLLg(inst, x, yStar);
    if isempty(gv)
        feasLL = true;
    else
        feasLL = all(gv <= 1e-6);
    end
    Fv = inst.F(x, yStar);
    Gv = evalG(inst, x, yStar);
end

%% ===================== STRONG STACKELBERG SELECTION (strong_stackelberg.py) =====================
% After every nested LL solve, the reply is replaced by the follower's tied optimal reply that is
% best for the leader (D'Amato et al. 2012), with the accuracy gate, validation, materiality and
% tie tests of strong_stackelberg.py. Local solves use fmincon (SQP) in MATLAB with the
% Optimization Toolbox, or sqp in Octave; without either, the step is skipped and the run follows
% the no-selection protocol (a warning is printed once).

function on = ssEnabled(set)
    persistent flag
    if isempty(flag), flag = true; end
    if nargin > 0, flag = logical(set); end
    on = flag;
end

function v = ssF(inst, x, y)
    llEvalCounter(1);   % LL objective evaluations made by the selection count in B1
    v = inst.f(x, y);
end

function v = ssViol(G)
    if isempty(G), v = 0; else, v = sum(max(G(:), 0) .^ 2); end
end

function [y, ok] = ssLocalMin(obj, cin, y0, lo, hi, maxit)
    % min obj(y) s.t. cin(y) <= 0, lo <= y <= hi, from y0. ok = false if no solver is available.
    persistent warned
    y0 = y0(:); lo = lo(:); hi = hi(:); ok = true;
    if exist('OCTAVE_VERSION', 'builtin') ~= 0
        h = @(v) -cin(v(:)');
        try
            y = sqp(y0, @(v) obj(v(:)'), [], h, lo, hi, maxit, 1e-12);
        catch
            y = y0;
        end
    elseif exist('fmincon', 'file') == 2 && license('test', 'optimization_toolbox')
        opts = optimoptions('fmincon', 'Algorithm', 'sqp', 'Display', 'off', 'MaxIterations', maxit, ...
                            'OptimalityTolerance', 1e-12, 'StepTolerance', 1e-14);
        try
            y = fmincon(@(v) obj(v(:)'), y0, [], [], [], [], lo, hi, @(v) deal(cin(v(:)'), []), opts);
        catch
            y = y0;
        end
    else
        if isempty(warned)
            warning('SA_APPA_2nd:ss', ['No fmincon/sqp available: strong Stackelberg selection skipped; ', ...
                    'results follow the original LL protocol.']);
            warned = true;
        end
        y = y0; ok = false; return;
    end
    y = min(max(y(:), lo), hi)';
end

function c = ssNoCon(~)
    c = -1;
end

function [y, f] = ssLocalLL(inst, x, y0, lo, hi, useG)
    if useG
        cin = @(v) reshape(evalLLg(inst, x, v), 1, []);
    else
        cin = @ssNoCon;
    end
    [y, ok] = ssLocalMin(@(v) ssF(inst, x, v), cin, y0, lo, hi, 200);
    if ~ok, y = y0; f = ssF(inst, x, y0); return; end
    f = ssF(inst, x, y);
    if useG && any(evalLLg(inst, x, y) > 1e-6)
        y = y0; f = ssF(inst, x, y0);
    end
end

function y = ssSelect(inst, x, y0)
    REL_EPS = 1e-8; ACC_TOL = 1e-5; DELTA_REL = 1e-4; TIE_RATIO = 0.5; FT = 1e-6;
    lo = inst.yb(:, 1)'; hi = inst.yb(:, 2)';
    y0 = min(max(y0(:)', lo), hi); y = y0;
    if ~ssEnabled(), return; end
    f0 = ssF(inst, x, y0); g0 = evalLLg(inst, x, y0);
    useG = ~isempty(g0) && all(g0 <= FT);
    if ~isempty(g0) && ~useG, return; end
    F0 = inst.F(x, y0); G0 = evalG(inst, x, y0); v0 = ssViol(G0);
    [~, fLoc] = ssLocalLL(inst, x, y0, lo, hi, useG);
    acc = ACC_TOL * max(1, abs(fLoc));
    if f0 - fLoc > acc, return; end                        % accuracy gate
    fRef = min(f0, fLoc);
    [c1, v1, F1] = ssPick(inst, x, y0, lo, hi, useG, fRef, REL_EPS * max(1, abs(fRef)), v0, G0, acc);
    if isempty(c1), return; end
    if v0 > FT^2, gainV = v0 - v1; else, gainV = 0; end
    if v1 <= max(v0, FT^2), gainF = F0 - F1; else, gainF = 0; end
    feasGain = gainV > FT^2 && (v1 <= FT^2 || gainV > DELTA_REL * max(1, v0));
    fGain = gainF > DELTA_REL * max(1, abs(F0));
    if ~(feasGain || fGain), return; end
    [c2, v2, F2] = ssPick(inst, x, y0, lo, hi, useG, fRef, REL_EPS * 1e-2 * max(1, abs(fRef)), v0, G0, acc);
    if isempty(c2), return; end
    if feasGain && (v0 - v2) < TIE_RATIO * gainV, feasGain = false; end   % tie test
    if fGain && (F0 - F2) < TIE_RATIO * gainF, fGain = false; end
    if feasGain || fGain, y = c2; end
end

function [best, bv, bF] = ssPick(inst, x, y0, lo, hi, useG, fRef, eps_, v0, G0, acc)
    FT = 1e-6;
    if useG
        base = @(v) [ssF(inst, x, v) - fRef - eps_, reshape(evalLLg(inst, x, v), 1, [])];
    else
        base = @(v) ssF(inst, x, v) - fRef - eps_;
    end
    start = y0; Gok = v0 <= FT^2;
    if ~Gok && ~isempty(G0)
        ya = ssLocalMin(@(v) ssViol(evalG(inst, x, v)), base, y0, lo, hi, 100);
        if ssViol(evalG(inst, x, ya)) < v0, start = ya; end
        Gok = ssViol(evalG(inst, x, start)) <= FT^2;
    end
    if Gok && ~isempty(G0)
        cons = @(v) [base(v), reshape(evalG(inst, x, v), 1, [])];
    else
        cons = base;
    end
    y1 = ssLocalMin(@(v) inst.F(x, v), cons, start, lo, hi, 100);
    best = []; bv = inf; bF = inf; cands = {y1, start};
    for k = 1:2
        c = cands{k};
        if isequal(c, y0), continue; end
        if ssF(inst, x, c) > fRef + 2 * eps_, continue; end
        if useG && any(evalLLg(inst, x, c) > FT), continue; end
        [~, fChk] = ssLocalLL(inst, x, c, lo, hi, useG);
        if fChk < fRef - acc, continue; end
        v = ssViol(evalG(inst, x, c)); Fc = inst.F(x, c);
        if isempty(best) || v < bv || (v == bv && Fc < bF)
            best = c; bv = v; bF = Fc;
        end
    end
end

%% ===================== THE 13 METHODS (methods.py) =====================
% Population-update engines. This file keeps Python's plain per-individual
% loops; random draws are taken in the same ORDER as methods.py (the draws
% themselves differ from numpy's generator -- see RANDOMNESS in the header).

function Xnew = appaStep(X, bestIdx, worstIdx, stream, lam)
    % Adaptive prey-predator update. lam fields: explore, local, evade, chase,
    % globl (the Python "global" coefficient), p_f.
    [N, d] = size(X);
    Xnew = X;
    pF = lam.p_f;
    for i = 1:N
        if i == worstIdx
            continue;
        end
        if rand(stream) <= pF
            r1 = rand(stream);
            diff = X(bestIdx, :) - X(i, :);
            nrm = norm(diff) + 1e-12;
            r2 = rand(stream, 1, d); %#ok<NASGU>  drawn and unused, as in methods.py
            Xnew(i, :) = X(i, :) + lam.explore * r1 * diff / nrm ...
                + lam.local * (rand(stream, 1, d) * 2 - 1);
        else
            r4 = rand(stream);
            diff = X(i, :) - X(worstIdx, :);
            nrm = norm(diff) + 1e-12;
            Xnew(i, :) = X(i, :) + lam.evade * r4 * diff / nrm;
        end
    end
    r5 = rand(stream);
    diff = X(bestIdx, :) - X(worstIdx, :);
    Xnew(worstIdx, :) = X(worstIdx, :) + lam.chase * r5 * diff + lam.globl * (rand(stream, 1, d) * 2 - 1);
end

function Xn = gaHybridize(X, fitness, stream, Pc, Pm, t, tMax, sigma0)
    % Crossover (+ occasional Gaussian mutation) offspring overwrite the worst
    % individuals, one per iteration, floor(N/3) iterations.
    [N, d] = size(X);
    [~, order] = sort(fitness(:));
    Xn = X;
    for it = 1:floor(N / 3)
        pair = randperm(stream, N, 2);
        i = pair(1); j = pair(2);
        if rand(stream) < Pc
            alpha = rand(stream, 1, d);
            off = alpha .* X(i, :) + (1 - alpha) .* X(j, :);
            if rand(stream) < Pm
                sigma = sigma0 * (1 - t / max(tMax, 1));
                off = off + sigma * randn(stream, 1, d);
            end
            worst = order(end);
            Xn(worst, :) = off;
            order = order(1:end-1);
        end
    end
end

function Xn = deStep(X, stream, F, CR)
    % DE/rand/1/bin generation: donors taken from the OLD population.
    [N, d] = size(X);
    Xn = X;
    for i = 1:N
        idxs = [1:i-1, i+1:N];
        pick = idxs(randperm(stream, N - 1, 3));
        a = X(pick(1), :); b = X(pick(2), :); c = X(pick(3), :);
        mutant = a + F * (b - c);
        cross = rand(stream, 1, d) < CR;
        if ~any(cross)
            cross(randi(stream, d)) = true;
        end
        trial = X(i, :);
        trial(cross) = mutant(cross);
        Xn(i, :) = trial;
    end
end

function [Xn, Vn] = psoStep(X, V, Pbest, gbest, stream, w, c1, c2)
    [N, d] = size(X);
    r1 = rand(stream, N, d);
    r2 = rand(stream, N, d);
    Vn = w * V + c1 * r1 .* (Pbest - X) + c2 * r2 .* (gbest - X);
    Xn = X + Vn;
end

function traj = trajNew()
    traj = struct('rows', zeros(0, 4), 'best', Inf, 'found', false);
end

function traj = trajRecord(traj, evalCount, iter, obj, feasible)
    % rows columns: eval_count, iteration, best_feasible_obj (NaN until the
    % first feasible point), is_feasible (cumulative "found a feasible point").
    if feasible && obj < traj.best
        traj.best = obj;
        traj.found = true;
    end
    if traj.found, b = traj.best; else, b = NaN; end
    traj.rows(end + 1, :) = [evalCount, iter, b, double(traj.found)];
end

function tf = isFeasibleG(Gv)
    tf = isempty(Gv) || all(Gv <= 1e-6);
end

function [X, Ftr, Gtr, feas] = initialDesign(inst, seed, initMode, cfg)
    % N0 initial upper-level points, each with an exact nested lower-level solve.
    N0 = cfg.N0;
    stream = newStream(seed);
    if strcmp(initMode, 'lhs')
        X = lhsSample(stream, inst.xb, N0);
    else
        X = uniformSample(stream, inst.xb, N0);
    end
    Ftr = zeros(N0, 1);
    feas = false(N0, 1);
    Gtr = zeros(N0, 0);
    for i = 1:N0
        [~, Fv, Gv, ~] = llSolve(inst, X(i, :), stableSeed(seed, 'init', i - 1), cfg.LL_POPSIZE, cfg.LL_MAXITER);
        Ftr(i) = Fv;
        if i == 1
            Gtr = zeros(N0, numel(Gv));
        end
        Gtr(i, :) = Gv;
        feas(i) = isFeasibleG(Gv);
    end
end

function [traj, cpuTime, snaps] = runAblationMethod(inst, methodId, engine, constraintMode, adaptation, initMode, seed, cfg, dgn)
    % One surrogate-assisted optimiser parameterised by engine ('appa'|'ga'|
    % 'de'|'pso'), constraint_mode ('cei'|'penalty'|'none'), adaptation
    % ('adaptive'|'fixed') and init ('lhs'|'uniform'). `dgn` (default []) is the
    % Appendix D.7 diagnostic override struct (pf_eta, pf_min, pf_max, beta,
    % gamma, delta); EVERY call from runMethod() leaves it empty, which is the
    % canonical untuned configuration (p_f held at 0.6, beta/gamma/delta =
    % 0.2/0.3/0.3), i.e. the Eq. 11 / p_f implementation gap is preserved.
    if nargin < 9, dgn = []; end
    % snaps (third output, requested only by phase7Figures) holds read-only copies of the
    % surrogates SA-APPA holds after the initial design and at the end of the run; asking for
    % it does not change the run.
    wantSnaps = nargout >= 3;
    snaps = struct();
    tStart = tic;
    N0 = cfg.N0; T_MAX = cfg.T_MAX; NPOP = cfg.N_UL_POP;
    gaHybrid = ~strcmp(methodId, 'M3');

    usePf = ~isempty(dgn) && isfield(dgn, 'pf_eta') && ~isempty(dgn.pf_eta);
    beta = 0.2; gamma = 0.3; delta = 0.3; pfMin = 0.1; pfMax = 0.9; pfEta = [];
    if ~isempty(dgn)
        if isfield(dgn, 'beta') && ~isempty(dgn.beta), beta = dgn.beta; end
        if isfield(dgn, 'gamma') && ~isempty(dgn.gamma), gamma = dgn.gamma; end
        if isfield(dgn, 'delta') && ~isempty(dgn.delta), delta = dgn.delta; end
        if isfield(dgn, 'pf_min') && ~isempty(dgn.pf_min), pfMin = dgn.pf_min; end
        if isfield(dgn, 'pf_max') && ~isempty(dgn.pf_max), pfMax = dgn.pf_max; end
        if usePf, pfEta = dgn.pf_eta; end
    end

    [X, Ftr, Gtr, feas] = initialDesign(inst, seed, initMode, cfg);
    traj = trajNew();
    evalCount = 0;
    for i = 1:N0
        evalCount = evalCount + 1;
        traj = trajRecord(traj, evalCount, 0, Ftr(i), feas(i));
    end
    if any(feas), fBest = min(Ftr(feas)); else, fBest = Inf; end

    if strcmp(methodId, 'M11'), surKind = 'rbf'; else, surKind = 'gp'; end
    mUl = size(Gtr, 2);
    [theta, phi] = fitSurrogates(X, Ftr, Gtr, mUl, surKind);
    if wantSnaps
        snaps.init = struct('X', X, 'F', Ftr, 'G', Gtr, 'theta', theta, 'phi', {phi});
    end

    streamPop = newStream(stableSeed(seed, methodId, inst.id, 'pop'));
    if strcmp(initMode, 'lhs')
        Pop = lhsSample(newStream(stableSeed(seed, methodId, inst.id, 'poplhs')), inst.xb, NPOP);
    else
        Pop = uniformSample(streamPop, inst.xb, NPOP);
    end
    V = zeros(size(Pop));
    Pbest = Pop;
    PbestVal = [];

    lam = struct('explore', 0.3, 'local', 0.1, 'evade', 0.3, 'chase', 0.3, 'globl', 0.2, 'p_f', 0.6);
    sigmaInit = std(Pop(:), 1);
    tauDiv = 0.3;
    dRange = inst.xb(:, 2) - inst.xb(:, 1);
    dMean = mean(dRange);

    for t = 0:T_MAX - 1
        % score the current population via the surrogate
        if strcmp(constraintMode, 'cei')
            [acq, mu, ~] = predictCEI(Pop, theta, phi, fBest);
        elseif strcmp(constraintMode, 'penalty')
            [acq, mu, ~] = predictPenalizedEI(Pop, theta, phi, fBest, t, T_MAX);
        else   % 'none' -> plain EI
            [mu, sg] = surrogatePredict(theta, Pop);
            acq = eiAcq(mu, sg, fBest);
        end
        bestIdx = argmaxNP(acq);
        worstIdx = argminNP(acq);
        fitnessGA = -acq;

        if strcmp(engine, 'appa')
            sigmaPop = std(Pop(:), 1);
            if strcmp(adaptation, 'adaptive')
                lam.explore = lam.explore * exp(beta * (tauDiv - sigmaPop / (sigmaInit + 1e-9)));
                deltaFit = (max(mu) - min(mu)) / (abs(max(mu)) + 1e-6);
                deltaFit = min(max(deltaFit, -1.0), 1.0);
                lam.evade = lam.evade * (1 - gamma * deltaFit);
                lam.chase = lam.chase * (1 + delta * deltaFit);
                lam.explore = min(max(lam.explore, 0.02), 1.0);
                lam.evade = min(max(lam.evade, 0.02), 1.0);
                lam.chase = min(max(lam.chase, 0.02), 1.0);
            else   % fixed schedule
                frac = 1 - t / max(T_MAX, 1);
                lam.explore = 0.3 * frac + 0.02;
                lam.evade = 0.3 * frac + 0.02;
                lam.chase = 0.3 * (1 - frac) + 0.02;
            end
            lamScaled = struct('explore', lam.explore * dMean, 'local', lam.local * dMean, ...
                'evade', lam.evade * dMean, 'chase', lam.chase * dMean, ...
                'globl', lam.globl * dMean, 'p_f', lam.p_f);
            Pop = appaStep(Pop, bestIdx, worstIdx, streamPop, lamScaled);
            if usePf && strcmp(adaptation, 'adaptive')
                % Eq. (11) p_f update -- DIAGNOSTIC ONLY (never active in the canonical runs).
                [muNew, ~] = surrogatePredict(theta, Pop);
                nSuccess = sum(muNew < mu);
                lam.p_f = min(max(lam.p_f + pfEta * (nSuccess / size(Pop, 1) - lam.p_f), pfMin), pfMax);
            end
            if gaHybrid && mod(t, 5) == 0
                rho = 0.3 * exp(-std(Pop(:), 1) / (sigmaInit + 1e-9));
                if rand(streamPop) < rho
                    Pop = gaHybridize(Pop, fitnessGA, streamPop, 0.8, 0.1, t, T_MAX, 0.15 * dMean);
                end
            end
        elseif strcmp(engine, 'ga')
            Pop = gaHybridize(Pop, fitnessGA, streamPop, 0.8, 0.2, t, T_MAX, 0.2 * dMean);
        elseif strcmp(engine, 'de')
            Pop = deStep(Pop, streamPop, 0.6, 0.8);
        elseif strcmp(engine, 'pso')
            if isempty(PbestVal)
                PbestVal = mu;
            end
            improved = mu < PbestVal;
            Pbest(improved, :) = Pop(improved, :);
            PbestVal(improved) = mu(improved);
            gbest = Pop(argminNP(mu), :);
            [Pop, V] = psoStep(Pop, V, Pbest, gbest, streamPop, 0.7, 1.5, 1.5);
        end

        Pop = clipRows(Pop, inst.xb);

        % infill: the current best-acquisition candidate gets a true evaluation
        if strcmp(constraintMode, 'cei')
            acq2 = predictCEI(Pop, theta, phi, fBest);
        elseif strcmp(constraintMode, 'penalty')
            acq2 = predictPenalizedEI(Pop, theta, phi, fBest, t, T_MAX);
        else
            [mu2, sg2] = surrogatePredict(theta, Pop);
            acq2 = eiAcq(mu2, sg2, fBest);
        end
        infillIdx = argmaxNP(acq2);
        xInfill = Pop(infillIdx, :);

        [~, Fv, Gv, ~] = llSolve(inst, xInfill, stableSeed(seed, methodId, inst.id, t), cfg.LL_POPSIZE, cfg.LL_MAXITER);
        feasX = isFeasibleG(Gv);
        evalCount = evalCount + 1;
        if feasX && Fv < fBest
            fBest = Fv;
        end
        traj = trajRecord(traj, evalCount, t + 1, Fv, feasX);

        X = [X; xInfill]; %#ok<AGROW>
        Ftr = [Ftr; Fv]; %#ok<AGROW>
        Gtr = [Gtr; Gv(:)']; %#ok<AGROW>
        [theta, phi] = fitSurrogates(X, Ftr, Gtr, numel(Gv), surKind);
    end
    if wantSnaps
        snaps.final = struct('X', X, 'F', Ftr, 'G', Gtr, 'theta', theta, 'phi', {phi});
    end
    cpuTime = toc(tStart);
end

function [traj, cpuTime] = runNestedMethod(inst, methodId, seed, cfg)
    % M12 and M13: the same nested GA with feasibility-rules ranking, differing
    % only in their seed streams (methodId enters every derived seed); no surrogate;
    % every population member gets an exact nested lower-level solve each
    % generation (NEST_POP x NEST_GEN = 60 expensive evaluations).
    tStart = tic;
    streamRng = newStream(stableSeed(seed, methodId, inst.id));
    X = lhsSample(newStream(stableSeed(seed, methodId, inst.id, 'init')), inst.xb, cfg.NEST_POP);
    traj = trajNew();
    evalCount = 0;
    dMean = mean(inst.xb(:, 2) - inst.xb(:, 1));

    [Fs, Gs, feass, traj, evalCount] = nestedEvaluate(inst, methodId, seed, cfg, X, 0, traj, evalCount);
    for gen = 1:cfg.NEST_GEN - 1
        % feasibility rules (Deb 2000): feasible-and-better beats infeasible;
        % among infeasible, lower total violation wins
        viol = zeros(size(Fs));
        for k = 1:numel(Fs)
            if ~isempty(Gs{k}), viol(k) = sum(max(Gs{k}, 0.0)); else, viol(k) = 0.0; end
        end
        rankKey = Fs;
        rankKey(~feass) = 1e12 + viol(~feass);
        [~, order] = sort(rankKey);
        Xr = X(order, :);
        Xnew = gaHybridize(Xr, rankKey(order), streamRng, 0.9, 0.15, gen, cfg.NEST_GEN, 0.2 * dMean);
        X = clipRows(Xnew, inst.xb);
        [Fs, Gs, feass, traj, evalCount] = nestedEvaluate(inst, methodId, seed, cfg, X, gen, traj, evalCount);
    end
    cpuTime = toc(tStart);
end

function [Fs, Gs, feass, traj, evalCount] = nestedEvaluate(inst, methodId, seed, cfg, Xpop, gen, traj, evalCount)
    n = size(Xpop, 1);
    Fs = zeros(n, 1); feass = false(n, 1); Gs = cell(n, 1);
    for i = 1:n
        [~, Fv, Gv, ~] = llSolve(inst, Xpop(i, :), stableSeed(seed, methodId, inst.id, gen, i - 1), ...
            cfg.LL_POPSIZE, cfg.LL_MAXITER);
        feasx = isFeasibleG(Gv);
        evalCount = evalCount + 1;
        traj = trajRecord(traj, evalCount, gen, Fv, feasx);
        Fs(i) = Fv; Gs{i} = Gv; feass(i) = feasx;
    end
end

function specs = methodSpecs()
    % Table 1 configurations (METHOD_SPECS in methods.py). M3 additionally
    % disables GA hybridisation inside runAblationMethod().
    mk = @(id, kind, eng, cm, ad, ini) struct('id', id, 'kind', kind, 'engine', eng, ...
        'constraintMode', cm, 'adaptation', ad, 'init', ini);
    specs = [ ...
        mk('M1', 'ablation', 'appa', 'cei', 'adaptive', 'lhs'), ...
        mk('M2', 'ablation', 'appa', 'cei', 'adaptive', 'uniform'), ...
        mk('M3', 'ablation', 'appa', 'cei', 'adaptive', 'lhs'), ...
        mk('M4', 'ablation', 'ga', 'cei', 'adaptive', 'lhs'), ...
        mk('M5', 'ablation', 'de', 'cei', 'adaptive', 'lhs'), ...
        mk('M6', 'ablation', 'pso', 'cei', 'adaptive', 'lhs'), ...
        mk('M7', 'ablation', 'appa', 'penalty', 'adaptive', 'lhs'), ...
        mk('M8', 'ablation', 'appa', 'none', 'adaptive', 'lhs'), ...
        mk('M9', 'ablation', 'appa', 'cei', 'fixed', 'lhs'), ...
        mk('M10', 'ablation', 'de', 'penalty', 'adaptive', 'lhs'), ...
        mk('M11', 'ablation', 'pso', 'penalty', 'adaptive', 'lhs'), ...
        mk('M12', 'nested', '', '', '', ''), ...
        mk('M13', 'nested', '', '', '', '')];
end

function [traj, cpuTime] = runMethod(inst, methodId, seed, cfg)
    specs = methodSpecs();
    k = find(strcmp({specs.id}, methodId), 1);
    if isempty(k), error('runMethod: unknown method id %s', methodId); end
    sp = specs(k);
    if strcmp(sp.kind, 'nested')
        [traj, cpuTime] = runNestedMethod(inst, methodId, seed, cfg);
    else
        [traj, cpuTime] = runAblationMethod(inst, methodId, sp.engine, sp.constraintMode, ...
            sp.adaptation, sp.init, seed, cfg, []);
    end
end

function [traj, cpuTime] = runM1_AppendixD7Diagnostic(inst, seed, cfg, diagCfg)
    % OPTIONAL, clearly labelled, NOT part of the default sweep (never called
    % from the main script body). M1 with the Appendix D.7 pf_eta-corrected
    % mechanism: p_f is updated each generation by Eq. (11) and the
    % beta/gamma/delta coefficients are overridden. The default values below
    % are the algorithm hyper-parameters of the single tuned re-run documented
    % in rerun_scenario5_tuned.py (configuration constants, not results).
    if nargin < 4 || isempty(diagCfg)
        diagCfg = struct('pf_eta', 0.055, 'beta', 0.379, 'gamma', 0.527, 'delta', 0.401, ...
            'pf_min', 0.1, 'pf_max', 0.9);
    end
    [traj, cpuTime] = runAblationMethod(inst, 'M1', 'appa', 'cei', 'adaptive', 'lhs', seed, cfg, diagCfg);
end

%% ===================== SWEEP (run_all.py) =====================

function R = runFullSweep(suite, methodIds, instanceIds, nRuns, cfg)
    % All methods x instances x runs, run serially, in the order method ->
    % instance -> run. Run r uses seed r (0..nRuns-1), as run_all.py does.
    % Checkpoints to sa_appa_2nd_output/sweep_checkpoint.mat every 100 runs.
    suiteIds = {suite.id};
    nM = numel(methodIds); nI = numel(instanceIds);
    instSuiteIdx = zeros(1, nI);
    for ii = 1:nI
        f = find(strcmp(suiteIds, instanceIds{ii}), 1);
        if isempty(f), error('runFullSweep: instance id "%s" not found in suite.', instanceIds{ii}); end
        instSuiteIdx(ii) = f;
    end
    nTotal = nM * nI * nRuns;
    mi = zeros(nTotal, 1); ii = zeros(nTotal, 1); rr = zeros(nTotal, 1);
    k = 0;
    for a = 1:nM
        for b = 1:nI
            for c = 0:nRuns - 1
                k = k + 1;
                mi(k) = a; ii(k) = b; rr(k) = c;
            end
        end
    end
    R = struct();
    R.methodIds = methodIds;
    R.instanceIds = instanceIds;
    R.mi = mi; R.ii = ii; R.run = rr;
    R.best = NaN(nTotal, 1);
    R.feas = false(nTotal, 1);
    R.cpu = NaN(nTotal, 1);
    R.rows = cell(nTotal, 1);
    R.nll = NaN(nTotal, 1);   % LL objective evaluations per run (Phase 7, B1)
    R.nTotal = nTotal;

    ckPath = fullfile('.', 'sa_appa_2nd_output', 'sweep_checkpoint.mat');
    ckKey = sprintf('%d|%s|%s', nTotal, strjoin(methodIds, ','), strjoin(instanceIds, ','));
    startIdx = 1;
    if exist(ckPath, 'file')
        try
            ck = load(ckPath);
            if isfield(ck, 'ckKey') && strcmp(ck.ckKey, ckKey)
                R.best = ck.best; R.feas = ck.feas; R.cpu = ck.cpu; R.rows = ck.rows;
                if isfield(ck, 'nll'), R.nll = ck.nll; end
                startIdx = ck.doneIdx + 1;
                fprintf('Resuming from checkpoint: %d / %d runs already done.\n', ck.doneIdx, nTotal);
            end
        catch
            startIdx = 1;
        end
    end

    tStart = tic;
    for idx = startIdx:nTotal
        methodId = methodIds{mi(idx)};
        inst = suite(instSuiteIdx(ii(idx)));
        try
            llEvalCounter(NaN);   % reset the LL evaluation counter
            [traj, cpuTime] = runMethod(inst, methodId, rr(idx), cfg);
            R.nll(idx) = llEvalCounter();
            if traj.found, R.best(idx) = traj.best; else, R.best(idx) = NaN; end
            R.feas(idx) = traj.found;
            R.cpu(idx) = cpuTime;
            R.rows{idx} = traj.rows;
        catch ME
            warning('runFullSweep: %s / %s run %d failed: %s', methodId, inst.id, rr(idx), ME.message);
            R.best(idx) = NaN; R.feas(idx) = false; R.cpu(idx) = NaN; R.rows{idx} = [];
        end
        if mod(idx, 25) == 0 || idx == nTotal
            elapsed = toc(tStart);
            done = idx - startIdx + 1;
            rate = done / max(elapsed, eps);
            etaH = (nTotal - idx) / max(rate, eps) / 3600;
            fprintf('[%s] %d/%d done, rate=%.3f/s, ETA=%.2f h\n', datestr(now, 'HH:MM:SS'), idx, nTotal, rate, etaH); %#ok<TNOW1,DATST>
        end
        if mod(idx, 100) == 0
            doneIdx = idx; best = R.best; feas = R.feas; cpu = R.cpu; rows = R.rows; nll = R.nll; %#ok<NASGU>
            try
                save(ckPath, 'ckKey', 'doneIdx', 'best', 'feas', 'cpu', 'rows', 'nll', '-v7');
            catch
            end
        end
    end
end

%% ===================== STATISTICS (analysis.py, friedman_test.py) =====================

function st = cellStats(R, m, i)
    sel = (R.mi == m) & (R.ii == i);
    nRuns = sum(sel);
    st = struct('n_runs', nRuns, 'n_feasible', 0, 'best', NaN, 'mean', NaN, 'std', NaN, 'cpu_mean', NaN);
    if nRuns == 0, return; end
    feasSel = sel & R.feas;
    vals = R.best(feasSel);
    st.n_feasible = numel(vals);
    if st.n_feasible > 0
        st.best = min(vals);
        st.mean = mean(vals);
        if st.n_feasible > 1, st.std = std(vals, 0); else, st.std = 0.0; end
    end
    st.cpu_mean = mean(R.cpu(sel));
end

function scen = scenarioDefs()
    mk = @(num, phase, name, methods, isolates) struct('num', num, 'phase', phase, 'name', name, ...
        'methods', {methods}, 'isolates', isolates);
    scen = [ ...
        mk(1, 'I', 'Initial design', {'M1', 'M2'}, 'LHS vs uniform initial design'), ...
        mk(2, 'I', 'GA hybridisation', {'M1', 'M3'}, 'with vs without periodic crossover and mutation'), ...
        mk(3, 'I', 'Search engine', {'M3', 'M4', 'M5', 'M6'}, ...
            'PPA vs GA, DE and PSO engines (M3, PPA without GA hybridisation, as the PPA representative)'), ...
        mk(4, 'I', 'Constraint handling', {'M1', 'M7', 'M8'}, 'CEI vs penalised EI vs unconstrained EI'), ...
        mk(5, 'I', 'Step-size rule', {'M1', 'M9'}, 'adaptive vs fixed step-size schedule'), ...
        mk(6, 'II', 'In-house surrogate-assisted engine baselines', {'M1', 'M10', 'M11'}, ...
            'SA-APPA vs SA-DE-Pen, SA-PSO-RBF'), ...
        mk(7, 'II', 'Fully nested baseline', {'M1', 'M12', 'M13'}, ...
            'SA-APPA vs nested GA (two seed streams)')];
end

function tbl = buildScenarioTable(R, sc)
    k = numel(sc.methods);
    nI = numel(R.instanceIds);
    tbl = struct('instanceIds', {R.instanceIds}, 'methods', {sc.methods});
    tbl.best = NaN(nI, k); tbl.mean = NaN(nI, k); tbl.std = NaN(nI, k);
    tbl.cpu = NaN(nI, k); tbl.nfeas = zeros(nI, k);
    for a = 1:k
        m = find(strcmp(R.methodIds, sc.methods{a}), 1);
        for b = 1:nI
            st = cellStats(R, m, b);
            tbl.best(b, a) = st.best; tbl.mean(b, a) = st.mean; tbl.std(b, a) = st.std;
            tbl.cpu(b, a) = st.cpu_mean; tbl.nfeas(b, a) = st.n_feasible;
        end
    end
end

function ranks = rankMatrixForScenario(meanM, nfeasM)
    % One row per instance, one column per method, rank 1 = best. Methods with
    % no feasible run on an instance are ranked worst (tied).
    [n, k] = size(meanM);
    ranks = zeros(n, k);
    for i = 1:n
        means = meanM(i, :);
        nf = nfeasM(i, :);
        valid = means(~isnan(means));
        if ~isempty(valid), BIG = max(valid) + 1e9; else, BIG = 1e9; end
        score = means;
        score(isnan(score)) = BIG;
        score(nf <= 0) = BIG;
        ranks(i, :) = rankAvg(score);
    end
end

function p = fUpperP(T, df1, df2)
    % Upper-tail probability of the F distribution (accurate for tiny p).
    p = betainc(df2 / (df2 + df1 * T), df2 / 2, df1 / 2);
end

function r = bisectRoot(fun, lo, hi, tol)
    flo = fun(lo); fhi = fun(hi);
    nExpand = 0;
    while sign(flo) == sign(fhi) && nExpand < 60
        hi = hi * 2; fhi = fun(hi); nExpand = nExpand + 1;
    end
    for it = 1:200
        mid = 0.5 * (lo + hi);
        fm = fun(mid);
        if sign(fm) == sign(flo)
            lo = mid; flo = fm;
        else
            hi = mid;
        end
        if (hi - lo) < tol * max(1, abs(mid)), break; end
    end
    r = 0.5 * (lo + hi);
end

function c = fCritical(df1, df2, alpha)
    c = bisectRoot(@(x) fUpperP(x, df1, df2) - alpha, 0, 10, 1e-12);
end

function t = tCritical(df, alpha)
    % Two-sided critical value t_{1-alpha/2, df}.
    tailFun = @(x) 0.5 * betainc(df / (df + x ^ 2), df / 2, 0.5) - alpha / 2;
    t = bisectRoot(tailFun, 0, 10, 1e-12);
end

function stat = friedmanStat(rankMat)
    [n, k] = size(rankMat);
    rankSums = sum(rankMat, 1);
    A2 = sum(rankMat(:) .^ 2);
    B2 = sum(rankSums .^ 2);
    C1 = n * k * (k + 1) ^ 2 / 4.0;
    SST = A2 - C1;
    SSt = B2 / n - C1;
    SSe = SST - SSt;
    df1 = k - 1; df2 = (n - 1) * (k - 1);
    if SSe <= 0
        T2 = NaN; pv = NaN;
    else
        T2 = (n - 1) * SSt / SSe;
        pv = fUpperP(T2, df1, df2);
    end
    stat = struct('n', n, 'k', k, 'rank_sums', rankSums, 'A2', A2, 'B2', B2, 'C1', C1, ...
        'SST', SST, 'SS_treatments', SSt, 'SS_error', SSe, 'T2', T2, 'df1', df1, 'df2', df2, 'p_value', pv);
end

function [cd, tcrit] = conoverCD(stat, alpha)
    % Conover post-hoc critical difference on |R_i - R_j| in RANK-SUM units.
    tcrit = tCritical(stat.df2, alpha);
    cd = tcrit * sqrt(2 * stat.n * stat.SS_error / stat.df2);
end

function P = srCdfInf(q, k)
    % CDF of the studentized range with infinite df (range of k standard normals).
    integrand = @(z) k * normPdf(z) .* (normCdf(z) - normCdf(z - q)) .^ (k - 1);
    P = integral(integrand, -Inf, Inf, 'RelTol', 1e-10, 'AbsTol', 1e-12);
end

function cd = nemenyiCD(k, n, alpha)
    % Nemenyi critical distance on MEAN ranks (distinct from the Conover CD).
    qAlpha = bisectRoot(@(q) srCdfInf(q, k) - (1 - alpha), 0.5, 10, 1e-10) / sqrt(2);
    cd = qAlpha * sqrt(k * (k + 1) / (6.0 * n));
end

function S = runScenario(R, sc)
    tbl = buildScenarioTable(R, sc);
    rankMat = rankMatrixForScenario(tbl.mean, tbl.nfeas);
    [n, k] = size(rankMat);
    stat = friedmanStat(rankMat);
    S = struct();
    S.scenario = sc;
    S.tbl = tbl;
    S.rankMat = rankMat;
    S.stat = stat;
    S.f_crit_05 = fCritical(stat.df1, stat.df2, 0.05);
    S.f_crit_01 = fCritical(stat.df1, stat.df2, 0.01);
    S.cd_conover_05 = conoverCD(stat, 0.05);
    S.cd_conover_01 = conoverCD(stat, 0.01);
    S.cd_nemenyi_05 = nemenyiCD(k, n, 0.05);
    S.meanRanks = mean(rankMat, 1);
end

function printScenarioReport(S)
    sc = S.scenario; st = S.stat;
    fprintf('%s\n', repmat('=', 1, 78));
    fprintf('Scenario %d (%s): %s  --  %s\n', sc.num, sc.phase, sc.name, strjoin(sc.methods, ', '));
    fprintf('Isolates: %s\n', sc.isolates);
    fprintf('n=%d instances, k=%d methods\n', st.n, st.k);
    fprintf('%s\n', repmat('-', 1, 78));
    fprintf('A2=%.3f  B2=%.3f  T2=%.4f  F(%d,%d)  p=%.4g\n', st.A2, st.B2, st.T2, st.df1, st.df2, st.p_value);
    if isnan(st.T2)
        fprintf('T2 is undefined here (SS_error <= 0: the rank pattern is identical across all blocks, e.g. very few instances).\n');
    end
    if st.T2 > S.f_crit_05, d05 = 'REJECT H0'; else, d05 = 'fail to reject'; end
    if st.T2 > S.f_crit_01, d01 = 'REJECT H0'; else, d01 = 'fail to reject'; end
    fprintf('F_crit(0.05)=%.3f -> %s\n', S.f_crit_05, d05);
    fprintf('F_crit(0.01)=%.3f -> %s\n', S.f_crit_01, d01);
    parts = cell(1, st.k);
    for a = 1:st.k, parts{a} = sprintf('%s=%.3f', sc.methods{a}, S.meanRanks(a)); end
    fprintf('Mean ranks: %s\n', strjoin(parts, ', '));
    fprintf('Conover post-hoc CD (rank-sum units): alpha=0.05 -> %.3f, alpha=0.01 -> %.3f\n', S.cd_conover_05, S.cd_conover_01);
    fprintf('Nemenyi CD (mean-rank units, for CD diagram): alpha=0.05 -> %.3f\n', S.cd_nemenyi_05);
    fprintf('%s\n', repmat('=', 1, 78));
end

function printScenarioTable(S)
    % Appendix B table: Best, Mean, Std, CPU Time (s) per method per instance.
    tbl = S.tbl; k = numel(tbl.methods);
    fprintf('\nAppendix B table, Scenario %d (Best / Mean / Std / CPU s per method)\n', S.scenario.num);
    hdr = sprintf('%-14s', 'Instance');
    for a = 1:k
        hdr = [hdr, sprintf('| %-4s %10s %10s %9s %7s ', tbl.methods{a}, 'Best', 'Mean', 'Std', 'CPU(s)')]; %#ok<AGROW>
    end
    fprintf('%s\n', hdr);
    for b = 1:numel(tbl.instanceIds)
        line = sprintf('%-14s', tbl.instanceIds{b});
        for a = 1:k
            line = [line, sprintf('| %-4s %10.4g %10.4g %9.4g %7.2f ', '', tbl.best(b, a), tbl.mean(b, a), tbl.std(b, a), tbl.cpu(b, a))]; %#ok<AGROW>
        end
        fprintf('%s\n', line);
    end
end

function s = csvNum(v, nd)
    if isnan(v)
        s = '';
    else
        s = sprintf('%.15g', round(v * 10 ^ nd) / 10 ^ nd);
    end
end

function writeScenarioCsv(S, path)
    tbl = S.tbl; k = numel(tbl.methods);
    fid = fopen(path, 'w');
    if fid < 0, warning('writeScenarioCsv: cannot open %s', path); return; end
    cols = {'instance_id'};
    for a = 1:k
        m = tbl.methods{a};
        cols = [cols, {[m '_best'], [m '_mean'], [m '_std'], [m '_cpu'], [m '_nfeas']}]; %#ok<AGROW>
    end
    fprintf(fid, '%s\n', strjoin(cols, ','));
    for b = 1:numel(tbl.instanceIds)
        parts = {tbl.instanceIds{b}};
        for a = 1:k
            parts = [parts, {csvNum(tbl.best(b, a), 4), csvNum(tbl.mean(b, a), 4), csvNum(tbl.std(b, a), 4), ...
                csvNum(tbl.cpu(b, a), 3), sprintf('%d', tbl.nfeas(b, a))}]; %#ok<AGROW>
        end
        fprintf(fid, '%s\n', strjoin(parts, ','));
    end
    fclose(fid);
end

function writeSummaryCsv(R, path)
    fid = fopen(path, 'w');
    if fid < 0, warning('writeSummaryCsv: cannot open %s', path); return; end
    fprintf(fid, 'method_id,instance_id,run_id,best,feasible_found,cpu_time_s\n');
    for k = 1:R.nTotal
        if R.feas(k), fs = 'True'; else, fs = 'False'; end
        if isnan(R.best(k)), bs = 'nan'; else, bs = sprintf('%.15g', R.best(k)); end
        fprintf(fid, '%s,%s,%d,%s,%s,%.10g\n', R.methodIds{R.mi(k)}, R.instanceIds{R.ii(k)}, R.run(k), bs, fs, R.cpu(k));
    end
    fclose(fid);
end

function validateFriedmanAgainstWorkedExample()
    % Formula check against the Friedmantest.pdf worked example (n = 21 blocks,
    % k = 4 treatments; A2 and B2 are given in the note itself). These inputs
    % are the note's published example, used only to validate the formulae.
    n = 21; k = 4;
    A2 = 630.0; B2 = 12266.0;
    C1 = n * k * (k + 1) ^ 2 / 4.0;
    SST = A2 - C1;
    SSt = B2 / n - C1;
    SSe = SST - SSt;
    T2 = (n - 1) * SSt / SSe;
    df1 = k - 1; df2 = (n - 1) * (k - 1);
    fc05 = fCritical(df1, df2, 0.05);
    fc01 = fCritical(df1, df2, 0.01);
    st = struct('n', n, 'k', k, 'SS_error', SSe, 'df2', df2);
    [cd01, t01] = conoverCD(st, 0.01);
    fprintf('    C1=%.0f (525)  SST=%.1f (105)  SS_treat=%.3f (59.095)  SS_err=%.3f (45.905)\n', C1, SST, SSt, SSe);
    fprintf('    T2=%.3f (25.74)  Fcrit(.05)=%.3f (2.76)  Fcrit(.01)=%.3f (4.13)\n', T2, fc05, fc01);
    fprintf('    t_crit(.01, df=60)=%.3f (2.660)  CD=%.3f (15.08)\n', t01, cd01);
    ok = abs(C1 - 525) < 1e-9 && abs(SST - 105) < 1e-6 && abs(SSt - 59.095) < 0.01 ...
        && abs(SSe - 45.905) < 0.01 && abs(T2 - 25.74) < 0.1 && abs(fc05 - 2.76) < 0.05 ...
        && abs(fc01 - 4.13) < 0.05 && abs(t01 - 2.660) < 0.01 && abs(cd01 - 15.08) < 0.1;
    if ~ok
        error('SA_APPA_2nd:friedmanSelfCheck', 'Friedman formula self-check FAILED -- see the printed values above.');
    end
    % Nemenyi q for k = 3 at alpha = 0.05 should be ~2.343 (Demsar 2006 table).
    qNem = bisectRoot(@(q) srCdfInf(q, 3) - 0.95, 0.5, 10, 1e-10) / sqrt(2);
    fprintf('    Nemenyi q(k=3, alpha=.05)=%.3f (2.343)\n', qNem);
    if abs(qNem - 2.343) > 0.01
        error('SA_APPA_2nd:nemenyiSelfCheck', 'Studentized-range quantile self-check FAILED.');
    end
    fprintf('    PASS.\n');
end

%% ===================== FIGURES (sa_appa_figures/figures/*.py) =====================
% All figures use opaque fills (no alpha), exported as EPS and PDF with the
% painters renderer. Multi-method plots use methodStyle() -- MATLAB's default
% colororder wraps after 7 colours, which would reproduce the colour-collision
% bug that once made M11's curve look like M1's in the Python figures.

function st = methodStyle(methodId, methodIds)
    % 13 distinct colours (the same assignment as figures/common.py's
    % _METHOD_COLORS) combined with a linestyle cycle, so no two methods in a
    % figure share a colour/linestyle pair.
    hexColors = {'#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', ...
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', ...
                 '#aec7e8', '#ffbb78', '#98df8a'};
    lineStyles = {'-', '--', '-.', ':'};
    idx = find(strcmp(methodIds, methodId), 1);
    if isempty(idx), idx = 1; end
    h = hexColors{mod(idx - 1, numel(hexColors)) + 1};
    st = struct();
    st.color = sscanf(h(2:end), '%2x')' / 255;
    st.linestyle = lineStyles{mod(idx - 1, numel(lineStyles)) + 1};
end

function c = lighten(color, amount)
    % Blend toward white at full opacity (no alpha) -- for the min/max bands.
    c = color + (1 - color) * amount;
end

function [fig, ax] = newFigure(w, h)
    fig = figure('Visible', 'off', 'Color', 'w', 'Units', 'inches', 'Position', [1 1 w h]);
    ax = axes('Parent', fig);
    hold(ax, 'on');
    box(ax, 'on');
    try
        % opaque, light-grey grid lines (no alpha anywhere in the EPS/PDF output)
        set(ax, 'GridAlpha', 1, 'GridColor', [0.85 0.85 0.85], 'MinorGridAlpha', 1);
    catch
    end
end

function addLegend(ax, handles, labels, loc, fs)
    try
        legend(ax, handles, labels, 'Location', loc, 'NumColumns', 2, 'FontSize', fs);
    catch
        try
            legend(ax, handles, labels, 'Location', loc, 'FontSize', fs);
        catch ME
            warning('addLegend: legend could not be drawn (%s); figure written without it.', ME.message);
        end
    end
end

function saveFigureBoth(fig, stem)
    set(fig, 'PaperUnits', 'inches');
    pos = get(fig, 'Position');
    set(fig, 'PaperSize', [pos(3) pos(4)], 'PaperPosition', [0 0 pos(3) pos(4)]);
    try
        print(fig, [stem '.eps'], '-depsc', '-painters');
    catch ME
        warning('saveFigureBoth: EPS export failed for %s: %s', stem, ME.message);
    end
    try
        print(fig, [stem '.pdf'], '-dpdf', '-painters');
    catch ME
        warning('saveFigureBoth: PDF export failed for %s: %s', stem, ME.message);
    end
    close(fig);
end

function drawBoxplot(ax, data, labels)
    % Hand-rolled boxplot (matplotlib semantics: whiskers to the furthest point
    % within 1.5 IQR, open-circle fliers, orange median, green triangle mean).
    medColor = [1.0 0.498 0.055];
    meanColor = [0.173 0.627 0.173];
    k = numel(data);
    for a = 1:k
        v = data{a}(:);
        v = v(~isnan(v));
        if isempty(v), continue; end
        q1 = percentileLocal(v, 25); med = percentileLocal(v, 50); q3 = percentileLocal(v, 75);
        iqr = q3 - q1;
        loFence = q1 - 1.5 * iqr; hiFence = q3 + 1.5 * iqr;
        inside = v(v >= loFence & v <= hiFence);
        loW = min(inside); hiW = max(inside);
        fl = v(v < loFence | v > hiFence);
        w = 0.25;
        patch('Parent', ax, 'XData', [a - w, a + w, a + w, a - w], 'YData', [q1, q1, q3, q3], ...
            'FaceColor', [1 1 1], 'EdgeColor', [0 0 0], 'LineWidth', 0.8);
        line('Parent', ax, 'XData', [a a], 'YData', [q3 hiW], 'Color', [0 0 0], 'LineWidth', 0.8);
        line('Parent', ax, 'XData', [a a], 'YData', [loW q1], 'Color', [0 0 0], 'LineWidth', 0.8);
        line('Parent', ax, 'XData', [a - w / 2, a + w / 2], 'YData', [hiW hiW], 'Color', [0 0 0], 'LineWidth', 0.8);
        line('Parent', ax, 'XData', [a - w / 2, a + w / 2], 'YData', [loW loW], 'Color', [0 0 0], 'LineWidth', 0.8);
        line('Parent', ax, 'XData', [a - w, a + w], 'YData', [med med], 'Color', medColor, 'LineWidth', 1.2);
        if ~isempty(fl)
            plot(ax, a * ones(size(fl)), fl, 'o', 'Color', [0 0 0], 'MarkerFaceColor', 'none', 'MarkerSize', 5);
        end
        plot(ax, a, mean(v), '^', 'Color', meanColor, 'MarkerFaceColor', meanColor, 'MarkerSize', 6);
    end
    set(ax, 'XLim', [0.4, k + 0.6], 'XTick', 1:k, 'XTickLabel', labels);
    try
        set(ax, 'XTickLabelRotation', 90);
    catch
    end
    grid(ax, 'on');
    set(ax, 'XGrid', 'off');
end

function [fL, finalMean] = bestAndMeans(R)
    % f_L(i): best final feasible value over all methods/runs on instance i.
    % finalMean(m,i): mean final feasible value of method m on instance i (NaN if none).
    nM = numel(R.methodIds); nI = numel(R.instanceIds);
    fL = NaN(1, nI);
    finalMean = NaN(nM, nI);
    for i = 1:nI
        sel = (R.ii == i) & R.feas;
        if any(sel), fL(i) = min(R.best(sel)); end
        for m = 1:nM
            s2 = sel & (R.mi == m);
            if any(s2), finalMean(m, i) = mean(R.best(s2)); end
        end
    end
end

function makeFigure1(R, figDir)
    % Figure 1, split into four standalone figures: TP1-TP9, SMD@(3,3),
    % SMD@(10,10), SMD@(30,30). Each run's final feasible objective is min-max
    % normalised per instance over every method and run; norm = 0 is the best
    % value found on that instance, 1 the worst.
    nI = numel(R.instanceIds);
    normVal = NaN(R.nTotal, 1);
    for i = 1:nI
        sel = (R.ii == i) & R.feas & ~isnan(R.best);
        if ~any(sel), continue; end
        lo = min(R.best(sel)); hi = max(R.best(sel));
        if hi > lo
            normVal(sel) = (R.best(sel) - lo) / (hi - lo);
        else
            normVal(sel) = 0.0;
        end
    end
    groupOf = cell(1, nI);
    for i = 1:nI
        tok = regexp(R.instanceIds{i}, '^SMD\d+_p(\d+)q(\d+)$', 'tokens', 'once');
        if isempty(tok), groupOf{i} = 'TP1-TP9'; else, groupOf{i} = sprintf('SMD @ (%s,%s)', tok{1}, tok{2}); end
    end
    groups = {'TP1-TP9', 'SMD @ (3,3)', 'SMD @ (10,10)', 'SMD @ (30,30)'};
    stems = {'fig1a_solution_by_method_TP1-TP9', 'fig1b_solution_by_method_SMD_3_3', ...
             'fig1c_solution_by_method_SMD_10_10', 'fig1d_solution_by_method_SMD_30_30'};
    for g = 1:numel(groups)
        inGroup = strcmp(groupOf, groups{g});
        if ~any(inGroup)
            fprintf('    fig1: no instances in group %s for this run, skipped.\n', groups{g});
            continue;
        end
        rowMask = inGroup(R.ii)';
        data = cell(1, numel(R.methodIds));
        for m = 1:numel(R.methodIds)
            v = normVal((R.mi == m) & rowMask(:));
            data{m} = v(~isnan(v));
        end
        [fig, ax] = newFigure(6.5, 5);
        drawBoxplot(ax, data, R.methodIds);
        set(ax, 'YLim', [-0.05, 1.05]);
        title(ax, ['Solution quality by method -- ' groups{g}], 'FontSize', 10, 'Interpreter', 'none');
        xlabel(ax, 'Method');
        ylabel(ax, sprintf('Min-max normalised final feasible objective\n(0 = best found, 1 = worst found, per instance)'));
        saveFigureBoth(fig, fullfile(figDir, stems{g}));
    end
end

function makeFigure2(R, figDir)
    % Dolan-More performance profile over all instances.
    [fL, finalMean] = bestAndMeans(R);
    valid = ~isnan(fL);
    nInst = sum(valid);
    tau = linspace(0, 2, 400);
    [fig, ax] = newFigure(7, 5);
    handles = zeros(1, numel(R.methodIds));
    for m = 1:numel(R.methodIds)
        e = (finalMean(m, valid) - fL(valid)) ./ (abs(fL(valid)) + 1e-8);
        e(isnan(e)) = Inf;
        rho = zeros(size(tau));
        for t = 1:numel(tau)
            rho(t) = mean(e <= tau(t));
        end
        st = methodStyle(R.methodIds{m}, R.methodIds);
        handles(m) = stairs(ax, tau, rho, 'Color', st.color, 'LineStyle', st.linestyle, 'LineWidth', 1.3);
    end
    xlabel(ax, '\tau (relative error threshold)');
    ylabel(ax, '\rho_m(\tau) = fraction of instances with e_{m,i} \leq \tau');
    set(ax, 'YLim', [-0.02 1.02]);
    title(ax, sprintf('Performance profile across %d reconciled instances', nInst));
    addLegend(ax, handles, R.methodIds, 'southeast', 7);
    text(0.02, 0.98, '''Best'' = empirical best found across all methods/runs; no certified optimum.', ...
        'Parent', ax, 'Units', 'normalized', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'top', ...
        'FontSize', 6, 'FontAngle', 'italic');
    grid(ax, 'on');
    saveFigureBoth(fig, fullfile(figDir, 'fig2_performance_profile'));
end

function makeFigure2b(R, suite, figDir, tau)
    % More-Wild data profile: fraction of (instance, run) pairs solved to
    % relative tolerance tau within a budget of kappa simplex gradients,
    % kappa = evals / (n_ul + 1).
    [fL, ~] = bestAndMeans(R);
    nI = numel(R.instanceIds);
    suiteIds = {suite.id};
    nUl = zeros(1, nI);
    for i = 1:nI, nUl(i) = suite(find(strcmp(suiteIds, R.instanceIds{i}), 1)).n_ul; end
    % f_0(i): mean best-feasible objective at eval_count == N0 over all methods/runs
    f0 = NaN(1, nI);
    for i = 1:nI
        vals = [];
        for k = find(R.ii == i)'
            rows = R.rows{k};
            if isempty(rows), continue; end
            r = find(rows(:, 1) == 25, 1);
            if ~isempty(r) && rows(r, 4) == 1, vals(end + 1) = rows(r, 3); end %#ok<AGROW>
        end
        if ~isempty(vals), f0(i) = mean(vals); end
    end
    kappaGrid = linspace(0, 25, 500);
    [fig, ax] = newFigure(7, 5);
    handles = zeros(1, numel(R.methodIds));
    for m = 1:numel(R.methodIds)
        kSolved = [];
        for k = find(R.mi == m)'
            rows = R.rows{k};
            if isempty(rows), continue; end
            i = R.ii(k);
            thr = fL(i) + tau * (f0(i) - fL(i));
            ok = rows(:, 4) == 1 & rows(:, 3) <= thr;
            r = find(ok, 1);
            if isempty(r), kSolved(end + 1) = Inf; else, kSolved(end + 1) = rows(r, 1) / (nUl(i) + 1); end %#ok<AGROW>
        end
        d = zeros(size(kappaGrid));
        for q = 1:numel(kappaGrid)
            d(q) = mean(kSolved <= kappaGrid(q));
        end
        st = methodStyle(R.methodIds{m}, R.methodIds);
        handles(m) = stairs(ax, kappaGrid, d, 'Color', st.color, 'LineStyle', st.linestyle, 'LineWidth', 1.3);
    end
    xlabel(ax, '\kappa (evaluation budget, simplex gradients = evals/(n_{UL}+1))');
    ylabel(ax, sprintf('d_m(\\kappa) = fraction solved to \\tau = %g', tau));
    set(ax, 'YLim', [-0.02 1.02]);
    title(ax, sprintf('Data profile (More and Wild, 2009), \\tau = %g', tau));
    addLegend(ax, handles, R.methodIds, 'southeast', 7);
    grid(ax, 'on');
    saveFigureBoth(fig, fullfile(figDir, sprintf('fig2b_data_profile_tau%g', tau)));
end

function Y = ffillColumns(Y)
    % Forward-fill NaNs down each column.
    for c = 1:size(Y, 2)
        last = NaN;
        for r = 1:size(Y, 1)
            if isnan(Y(r, c)), Y(r, c) = last; else, last = Y(r, c); end
        end
    end
end

function makeFigure3One(R, instanceId, figDir)
    % Mean best-feasible-so-far vs expensive evaluations, one curve per method,
    % opaque lightened min-max band across runs. All bands are drawn first and
    % all lines afterwards, so no band can hide another method's curve.
    i = find(strcmp(R.instanceIds, instanceId), 1);
    if isempty(i), return; end
    [fig, ax] = newFigure(6.5, 4.5);
    omitted = {};
    curves = struct('id', {}, 'm', {}, 'x', {}, 'mu', {}, 'lo', {}, 'hi', {}, 'ok', {});
    for m = 1:numel(R.methodIds)
        runs = find(R.mi == m & R.ii == i)';
        nE = 0;
        for k = runs, if ~isempty(R.rows{k}), nE = max(nE, size(R.rows{k}, 1)); end, end
        if nE == 0, omitted{end + 1} = R.methodIds{m}; continue; end %#ok<AGROW>
        Y = NaN(nE, numel(runs));
        for c = 1:numel(runs)
            rows = R.rows{runs(c)};
            if ~isempty(rows), Y(1:size(rows, 1), c) = rows(:, 3); end
        end
        Y = ffillColumns(Y);
        cnt = sum(~isnan(Y), 2);
        if all(cnt == 0), omitted{end + 1} = R.methodIds{m}; continue; end %#ok<AGROW>
        Y0 = Y; Y0(isnan(Y0)) = 0;
        mu = sum(Y0, 2) ./ cnt; mu(cnt == 0) = NaN;
        curves(end + 1) = struct('id', R.methodIds{m}, 'm', m, 'x', (1:nE)', 'mu', mu, ...
            'lo', min(Y, [], 2), 'hi', max(Y, [], 2), 'ok', cnt > 0); %#ok<AGROW>
    end
    for q = 1:numel(curves)
        c = curves(q);
        st = methodStyle(c.id, R.methodIds);
        patch('Parent', ax, 'XData', [c.x(c.ok); flipud(c.x(c.ok))], 'YData', [c.lo(c.ok); flipud(c.hi(c.ok))], ...
            'FaceColor', lighten(st.color, 0.78), 'EdgeColor', 'none');
    end
    handles = zeros(1, numel(curves)); labels = cell(1, numel(curves));
    for q = 1:numel(curves)
        c = curves(q);
        st = methodStyle(c.id, R.methodIds);
        handles(q) = plot(ax, c.x, c.mu, 'Color', st.color, 'LineStyle', st.linestyle, 'LineWidth', 1.4);
        labels{q} = c.id;
    end
    xlabel(ax, 'Cumulative expensive evaluations');
    ylabel(ax, 'Best feasible objective found so far');
    title(ax, ['Convergence on ' instanceId], 'Interpreter', 'none');
    if ~isempty(handles)
        addLegend(ax, handles, labels, 'best', 6);
    end
    if ~isempty(omitted)
        text(0.02, 0.02, ['No feasible runs: ' strjoin(omitted, ', ')], 'Parent', ax, 'Units', 'normalized', 'FontSize', 6, 'Color', [0.5 0.5 0.5]);
    end
    grid(ax, 'on');
    saveFigureBoth(fig, fullfile(figDir, ['fig3_convergence_' instanceId]));
end

function [order, nPts] = fitConvergenceOrder(it, err, zeroFloor)
    % OLS fit of log(error) on log(iteration) over the informative portion;
    % order = -slope (error ~ t^-order). Returns NaN, 0 with fewer than 3 points.
    if nargin < 3, zeroFloor = 1e-10; end
    order = NaN; nPts = 0;
    mask = (it >= 1) & (err > zeroFloor) & isfinite(err);
    if sum(mask) < 3, return; end
    x = log(it(mask)); y = log(err(mask));
    fin = isfinite(x) & isfinite(y);
    x = x(fin); y = y(fin);
    if numel(x) < 3 || (max(x) - min(x)) == 0, return; end
    p = polyfit(x(:), y(:), 1);
    order = -p(1);
    nPts = numel(x);
end

function makeFigure4One(R, instanceId, figDir)
    % Log-log |f_t - f_best,run| vs iteration, one curve per method (mean over
    % runs), legend reports the fitted empirical order (mean +/- std over runs).
    i = find(strcmp(R.instanceIds, instanceId), 1);
    if isempty(i), return; end
    [fig, ax] = newFigure(6.5, 4.5);
    handles = []; labels = {};
    for m = 1:numel(R.methodIds)
        runs = find(R.mi == m & R.ii == i)';
        orders = []; E = []; itCol = [];
        for k = runs
            rows = R.rows{k};
            if isempty(rows), continue; end
            vals = ffillColumns(rows(:, 3));
            if all(isnan(vals)), continue; end
            err = abs(vals - vals(end));
            [ord, npts] = fitConvergenceOrder(rows(:, 2), err, 1e-10);
            if npts >= 3, orders(end + 1) = ord; end %#ok<AGROW>
            if isempty(E), E = NaN(size(rows, 1), 0); itCol = rows(:, 2); end
            if size(E, 1) == numel(err), E(:, end + 1) = err; end %#ok<AGROW>
        end
        st = methodStyle(R.methodIds{m}, R.methodIds);
        if isempty(E)
            continue;
        end
        E0 = E; E0(isnan(E0)) = 0;
        cnt = sum(~isnan(E), 2);
        meanErr = sum(E0, 2) ./ cnt; meanErr(cnt == 0) = NaN;
        keep = itCol >= 1 & meanErr > 1e-10 & isfinite(meanErr);
        if isempty(orders)
            lab = sprintf('%s (order n/a)', R.methodIds{m});
        else
            lab = sprintf('%s (p=%.2f+/-%.2f)', R.methodIds{m}, mean(orders), std(orders, 1));
        end
        if sum(keep) >= 2
            h = loglog(ax, itCol(keep), meanErr(keep), 'Color', st.color, 'LineStyle', st.linestyle, 'LineWidth', 1.3);
            handles(end + 1) = h; labels{end + 1} = lab; %#ok<AGROW>
        end
    end
    xlabel(ax, 'Iteration t (log scale)');
    ylabel(ax, '|f_t - f_{best,run}| (log scale)');
    title(ax, ['Empirical convergence rate on ' instanceId], 'Interpreter', 'none');
    if ~isempty(handles)
        addLegend(ax, handles, labels, 'best', 6);
    end
    grid(ax, 'on');
    saveFigureBoth(fig, fullfile(figDir, ['fig4_convergence_rate_' instanceId]));
end

function makeCdDiagram(S, figDir)
    % Critical-difference diagram from MATLAB's own Friedman run: methods placed
    % by mean rank, bars join methods closer than the Nemenyi CD. Opaque bars.
    sc = S.scenario;
    methods = sc.methods;
    k = numel(methods);
    ranks = S.meanRanks;
    cd = S.cd_nemenyi_05;
    [rs, ord] = sort(ranks);
    order = methods(ord);
    [fig, ax] = newFigure(8, 2.6 + 0.2 * k);
    y0 = 0.5;
    line('Parent', ax, 'XData', [min(rs) - 0.5, max(rs) + 0.5], 'YData', [y0 y0], 'Color', [0 0 0], 'LineWidth', 1);
    for i = 1:k
        r = rs(i);
        line('Parent', ax, 'XData', [r r], 'YData', [y0, y0 + 0.15], 'Color', [0 0 0], 'LineWidth', 1);
        text(r, y0 + 0.18 + 0.26 * mod(i - 1, 2), sprintf('%s\n(%.2f)', order{i}, r), ...
            'Parent', ax, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 8);
    end
    barY = y0 - 0.15;
    i = 1;
    while i <= k
        j = i;
        while j + 1 <= k && rs(j + 1) - rs(i) < cd
            j = j + 1;
        end
        if j > i
            line('Parent', ax, 'XData', [rs(i) rs(j)], 'YData', [barY barY], 'Color', [0.698 0.133 0.133], 'LineWidth', 3);
            barY = barY - 0.1;
        end
        if j == i, i = i + 1; else, i = i + (j - i); end
    end
    set(ax, 'YLim', [barY - 0.1, 0.5 + 0.18 + 0.26 + 0.55], 'YTick', [], 'YColor', 'none', ...
        'XLim', [min(rs) - 0.6, max(rs) + 0.6], 'Box', 'off');
    xlabel(ax, 'Mean rank (lower = better)');
    title(ax, sprintf('Scenario %d: %s  (CD=%.3f, \\alpha=0.05)', sc.num, sc.name, cd));
    saveFigureBoth(fig, fullfile(figDir, sprintf('fig_cd_scenario%d', sc.num)));
end

%% ===================== SELF-VALIDATION REPORT =====================
% The Python scenario tables are read HERE ONLY, as a static reference for
% comparison. They are never a data source for MATLAB's own runs, tables or
% figures. The Friedman statistic for the reference is recomputed from the
% reference tables with THIS file's own (self-checked) Friedman routines.

function py = loadPythonScenarioTable(n)
    py = struct('ok', false, 'path', '', 'instanceIds', {{}}, 'methods', {{}}, 'mean', [], 'nfeas', []);
    rel = fullfile('scenario_tables', sprintf('scenario_%d.csv', n));
    candidates = { ...
        fullfile('sa_appa_codebase', 'sa_appa_experiments', 'results', rel), ...
        fullfile('sa_appa_experiments', 'results', rel), ...
        fullfile('results', rel), ...
        rel, ...
        fullfile('..', 'sa_appa_experiments', 'results', rel), ...
        fullfile('..', 'sa_appa_codebase', 'sa_appa_experiments', 'results', rel)};
    found = '';
    for c = 1:numel(candidates)
        if exist(candidates{c}, 'file') == 2, found = candidates{c}; break; end
    end
    if isempty(found), return; end
    fid = fopen(found, 'r');
    if fid < 0, return; end
    hdr = strsplit(strtrim(fgetl(fid)), ',');
    lines = {};
    ln = fgetl(fid);
    while ischar(ln)
        if ~isempty(strtrim(ln)), lines{end + 1} = ln; end %#ok<AGROW>
        ln = fgetl(fid);
    end
    fclose(fid);
    nRows = numel(lines);
    vals = cell(nRows, numel(hdr));
    for r = 1:nRows
        parts = strsplit(lines{r}, ',', 'CollapseDelimiters', false);
        for c = 1:numel(hdr)
            if c <= numel(parts), vals{r, c} = parts{c}; else, vals{r, c} = ''; end
        end
    end
    methods = {};
    for c = 2:numel(hdr)
        tok = regexp(hdr{c}, '^(M\d+)_mean$', 'tokens', 'once');
        if ~isempty(tok), methods{end + 1} = tok{1}; end %#ok<AGROW>
    end
    py.instanceIds = vals(:, 1)';
    py.methods = methods;
    py.mean = NaN(nRows, numel(methods));
    py.nfeas = zeros(nRows, numel(methods));
    for a = 1:numel(methods)
        cm = find(strcmp(hdr, [methods{a} '_mean']), 1);
        cn = find(strcmp(hdr, [methods{a} '_nfeas']), 1);
        for r = 1:nRows
            py.mean(r, a) = str2double(vals{r, cm});
            py.nfeas(r, a) = str2double(vals{r, cn});
        end
    end
    py.path = found;
    py.ok = true;
end

function validationReport(S, scen, instanceIds, quick)
    fprintf('\n%s\n', repmat('=', 1, 100));
    fprintf('SELF-VALIDATION: MATLAB (this run) vs Python reference scenario tables\n');
    fprintf('Agreement means: same Friedman decision at alpha = 0.05 and 0.01, and the best mean rank\n');
    fprintf('is held by the same method (a tie for best counts as agreement if the sets overlap).\n');
    if quick
        fprintf('QUICK_TEST is on: only %d instances x 3 runs. The Friedman tests below have n = %d blocks,\n', numel(instanceIds), numel(instanceIds));
        fprintf('so this table is a smoke test of the code path, NOT a validation of the translation.\n');
    end
    fprintf('%s\n', repmat('-', 1, 100));
    fprintf('%-3s %-4s | %-22s | %-22s | %-9s %-9s | %-12s %-12s | %s\n', 'Sc', 'n', ...
        'MATLAB T2 (p)', 'Python-ref T2 (p)', 'rej.05 M/P', 'rej.01 M/P', 'best M', 'best P', 'verdict');
    nAgree = 0; nDisagree = 0; nSkipped = 0;
    for s = 1:numel(scen)
        sc = scen(s); M = S{s};
        py = loadPythonScenarioTable(sc.num);
        if ~py.ok
            fprintf('%-3d %-4s | Python reference table scenario_%d.csv not found -- skipped\n', sc.num, '-', sc.num);
            nSkipped = nSkipped + 1;
            continue;
        end
        % align the Python reference to the instances MATLAB actually ran
        [~, ia, ib] = intersect(M.tbl.instanceIds, py.instanceIds, 'stable');
        if numel(ia) < 3
            fprintf('%-3d %-4d | fewer than 3 shared instances with the reference -- skipped\n', sc.num, numel(ia));
            nSkipped = nSkipped + 1;
            continue;
        end
        cols = zeros(1, numel(sc.methods));
        for a = 1:numel(sc.methods)
            cols(a) = find(strcmp(py.methods, sc.methods{a}), 1);
        end
        rankPy = rankMatrixForScenario(py.mean(ib, cols), py.nfeas(ib, cols));
        statPy = friedmanStat(rankPy);
        fc05 = fCritical(statPy.df1, statPy.df2, 0.05);
        fc01 = fCritical(statPy.df1, statPy.df2, 0.01);
        % MATLAB statistic recomputed on the same instance subset (identical to S{s} when all 44 ran)
        rankMl = M.rankMat(ia, :);
        statMl = friedmanStat(rankMl);
        rejMl05 = statMl.T2 > fCritical(statMl.df1, statMl.df2, 0.05);
        rejMl01 = statMl.T2 > fCritical(statMl.df1, statMl.df2, 0.01);
        rejPy05 = statPy.T2 > fc05; rejPy01 = statPy.T2 > fc01;
        mrMl = mean(rankMl, 1); mrPy = mean(rankPy, 1);
        bestMl = sc.methods(abs(mrMl - min(mrMl)) < 1e-9);
        bestPy = sc.methods(abs(mrPy - min(mrPy)) < 1e-9);
        sameDecision = (rejMl05 == rejPy05) && (rejMl01 == rejPy01);
        sameBest = ~isempty(intersect(bestMl, bestPy));
        if sameDecision && sameBest
            verdict = 'AGREE'; nAgree = nAgree + 1;
        else
            verdict = 'DISAGREE'; nDisagree = nDisagree + 1;
        end
        fprintf('%-3d %-4d | %9.3f (%-10.3g) | %9.3f (%-10.3g) | %-4s/%-4s %-4s/%-4s | %-12s %-12s | %s\n', ...
            sc.num, numel(ia), statMl.T2, statMl.p_value, statPy.T2, statPy.p_value, ...
            yn(rejMl05), yn(rejPy05), yn(rejMl01), yn(rejPy01), strjoin(bestMl, '+'), strjoin(bestPy, '+'), verdict);
    end
    fprintf('%s\n', repmat('-', 1, 100));
    fprintf('Summary: %d AGREE, %d DISAGREE, %d skipped (of %d scenarios).\n', nAgree, nDisagree, nSkipped, numel(scen));
    if nDisagree > 0
        fprintf('DISAGREEMENT is reported as found: a disagreement is a translation bug (or a statistically\n');
        fprintf('plausible difference between two different random-number generators) to investigate, not to\n');
        fprintf('paper over. Compare the mean ranks and the per-instance tables in %s.\n', fullfile('sa_appa_2nd_output', 'scenario_tables'));
    end
    if nSkipped == numel(scen)
        fprintf('No Python reference tables were found. Place scenario_tables/scenario_{1..7}.csv (from\n');
        fprintf('sa_appa_experiments/results/) next to this script to enable the comparison.\n');
    end
    fprintf('%s\n', repmat('=', 1, 100));
end

function s = yn(tf)
    if tf, s = 'Y'; else, s = 'N'; end
end

%% ===================== END OF FILE =====================

%% ===================== PHASE 7 EXTRA FIGURES (figures_phase7.py, phase7_data.py) =====================
% Drawn only when MAKE_EXTRA_FIGS = true. Same logic as the Python files
% sa_appa_figures/figures_phase7.py and sa_appa_experiments/phase7_data.py /
% phase7_run.py / instrument_ll.py, applied to THIS file's own sweep results:
%   Part A  surrogate fidelity on TP3 (and the UL constraint maps of the
%           low-dimensional TPs). SA-APPA's GPs model theta(x) = F(x, y*(x)) and
%           the UL constraint maps phi_j(x); nothing is drawn or described as an
%           approximation of the reaction set, of y*(x) or of the LL value
%           function f*(x).
%   Part B  B1 budget composition (UL evaluations + LL objective evaluations,
%           counted by llEvalCounter() during the sweep, all runs) and B2
%           evaluations to target (from the stored trajectories, all runs).
% Every figure is its own file (EPS + PDF, opaque: wireframes, hatching, no
% alpha). Configuration M14 (BL-SAEA) exists only in the Python code, so the
% Phase II group here is M1, M10-M13.

function n = llEvalCounter(delta)
    % Running count of LL objective evaluations. llEvalCounter(1) adds one,
    % llEvalCounter(NaN) resets, llEvalCounter() reads.
    persistent cnt
    if isempty(cnt), cnt = 0; end
    if nargin > 0
        if isnan(delta), cnt = 0; else, cnt = cnt + delta; end
    end
    n = cnt;
end

function o = phase7Options(quick)
    o = struct();
    o.nGrid = 61;            % wireframe / error-map grid per axis
    o.nHeldTP3 = 2500;       % held-out LHS points for the TP3 fidelity table
    o.nHeldTP = 400;         % held-out points per low-dimensional TP (A5)
    o.nHeldSMD = 200;        % held-out points per SMD (3,3) instance (A5)
    o.rNear = 1.5;           % radius of the neighbourhood of x* (10% of the TP3 box)
    o.heldSeed = 20261007;   % seed of every held-out set (fixed in advance)
    o.refPop = 30;           % reference LL solve: popsize multiplier
    o.refIter = 300;         %                     generations (best of two seeds)
    o.targets = [1e-2, 1.0]; % B2 absolute targets (1e-2 all sets; 1.0 SMD, supplementary)
    if quick
        o.nGrid = 31; o.nHeldTP3 = 300; o.nHeldTP = 40; o.nHeldSMD = 20; o.refIter = 60;
    end
end

function st = phase7Style(m)
    % Colour and line style as in the manuscript figures (figures_paper.STYLE),
    % plus a marker and a hatch kind per configuration for greyscale printing.
    P = {'#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948'};
    GRAY = '#52514e';
    tbl = { ...
        'M1', P{1}, '-', 'o', 'none'; 'M2', P{2}, '-', 's', 'fwd'; 'M3', P{3}, '-', '^', 'bwd'; ...
        'M4', P{4}, '-', 'd', 'cross'; 'M5', P{5}, '-', 'v', 'dots'; 'M6', P{6}, '-', 'p', 'horiz'; ...
        'M7', P{7}, '-', 'x', 'plus'; 'M8', P{8}, '-', '<', 'vert'; 'M9', GRAY, '--', '>', 'dots'; ...
        'M10', P{2}, '-', 's', 'bwd'; 'M11', P{3}, '-', '^', 'fwd'; 'M12', P{4}, '-', 'd', 'horiz'; ...
        'M13', P{4}, '--', 'h', 'vert'};
    k = find(strcmp(tbl(:, 1), m), 1);
    if isempty(k), k = 1; end
    h = tbl{k, 2};
    st = struct('color', sscanf(h(2:end), '%2x')' / 255, 'linestyle', tbl{k, 3}, ...
        'marker', tbl{k, 4}, 'hatch', tbl{k, 5});
end

function lab = phase7Label(m)
    names = {'M1', 'M1 SA-APPA'; 'M2', 'M2 uniform init.'; 'M3', 'M3 no GA'; 'M4', 'M4 GA engine'; ...
        'M5', 'M5 DE engine'; 'M6', 'M6 PSO engine'; 'M7', 'M7 penalised EI'; 'M8', 'M8 plain EI'; ...
        'M9', 'M9 fixed schedule'; 'M10', 'M10 SA-DE-Pen'; 'M11', 'M11 SA-PSO-RBF'; ...
        'M12', 'M12 Nest-GA (A)'; 'M13', 'M13 Nest-GA (B)'};
    k = find(strcmp(names(:, 1), m), 1);
    if isempty(k), lab = m; else, lab = names{k, 2}; end
end

function inst = phase7Inst(suite, id)
    k = find(strcmp({suite.id}, id), 1);
    if isempty(k), inst = []; else, inst = suite(k); end
end

% ---------------------------------------------------------------- ground truth
function [y, exact] = phase7Ystar(inst, x, o)
    % TP3 and TP8: the LL problem is a strictly convex QP with simple bounds;
    % its solution is the exact projection below. Others: reference LL solve
    % (this file's DE, popsize o.refPop * n_ll, o.refIter generations, polish,
    % best of two seeds) -- a heuristic, described as such.
    exact = true;
    if strcmp(inst.id, 'TP3')
        y = min(max(x, 0), 10);
    elseif strcmp(inst.id, 'TP8')
        hi = min(20, (x - 10) / 2);
        y = min(max(x - 20, -10), hi);
    else
        exact = false;
        [y1, ~, ~, ~] = llSolve(inst, x, 0, o.refPop, o.refIter);
        [y2, ~, ~, ~] = llSolve(inst, x, 1, o.refPop, o.refIter);
        if llObjective(inst, x, y2) < llObjective(inst, x, y1), y = y2; else, y = y1; end
    end
end

function T = phase7Truth(inst, x, o)
    [y, exact] = phase7Ystar(inst, x, o);
    g = evalLLg(inst, x, y);
    T = struct('y', y, 'theta', inst.F(x, y), 'phi', evalG(inst, x, y), 'fstar', inst.f(x, y), ...
        'llFeasible', isempty(g) || all(g <= 1e-6), 'exact', exact);
end

% ---------------------------------------------------------------- run selection / snapshots
function [runId, idx] = phase7MedianRun(R, methodId, instId)
    % Rule fixed in advance: sort the method's runs on the instance by final
    % best feasible UL objective (runs without a feasible point last), ties by
    % run id, and take the lower median (the 15th of 30).
    mI = find(strcmp(R.methodIds, methodId)); iI = find(strcmp(R.instanceIds, instId));
    sel = find(R.mi == mI & R.ii == iI);
    key = R.best(sel); key(~R.feas(sel)) = Inf;
    [~, ord] = sortrows([key, R.run(sel)]);
    pick = sel(ord(ceil(numel(sel) / 2)));
    runId = R.run(pick); idx = pick;
end

function [snaps, chk] = phase7Rerun(R, inst, methodId, runId, idx, cfg)
    specs = methodSpecs();
    sp = specs(strcmp({specs.id}, methodId));
    [traj, ~, snaps] = runAblationMethod(inst, methodId, sp.engine, sp.constraintMode, ...
        sp.adaptation, sp.init, runId, cfg, []);
    a = traj.rows(:, 3); b = R.rows{idx}(:, 3);
    if numel(a) ~= numel(b)
        dev = Inf;
    else
        bothNaN = isnan(a) & isnan(b);
        d = abs(a - b) ./ max(1, abs(b)); d(bothNaN) = 0; d(xor(isnan(a), isnan(b))) = Inf;
        dev = max([d; 0]);
    end
    chk = struct('instance', inst.id, 'method', methodId, 'run', runId, 'maxRelDev', dev, ...
        'reproduces', dev <= 1e-6);
end

function M = phase7Metrics(model, Xh, th)
    [mu, sd] = surrogatePredict(model, Xh);
    e = mu - th(:);
    M = struct('n', numel(th), 'rmse', sqrt(mean(e .^ 2)), 'maxAbs', max(abs(e)), ...
        'coverage', mean(abs(e) <= 2 * sd), 'meanSd', mean(sd), 'range', max(th) - min(th));
end

% ---------------------------------------------------------------- drawing helpers
function [fig, ax] = phase7Fig3d()
    fig = figure('Visible', 'off', 'Color', 'w', 'Units', 'inches', 'Position', [1 1 4.6 3.8]);
    ax = axes('Parent', fig); hold(ax, 'on'); grid(ax, 'on');
    try, set(ax, 'GridAlpha', 1, 'GridColor', [0.89 0.89 0.87]); catch, end
    view(ax, -35, 26);
    xlabel(ax, 'x_1'); ylabel(ax, 'x_2');
end

function h = phase7Wire(ax, X1, X2, Z, color, lw, ls, stride)
    r = 1:stride:size(X1, 1);
    h = mesh(ax, X1(r, r), X2(r, r), Z(r, r), 'FaceColor', 'none', 'EdgeColor', color, ...
        'LineWidth', lw, 'LineStyle', ls);
end

function cmap = phase7Seq(n)
    stops = [0.957 0.973 0.992; 0.612 0.765 0.937; 0.165 0.471 0.839; 0.071 0.243 0.451];
    t = linspace(0, 1, size(stops, 1));
    cmap = interp1(t, stops, linspace(0, 1, n));
end

function phase7Save(fig, figDir, stem, caption, fidCap)
    saveFigureBoth(fig, fullfile(figDir, stem));
    fprintf(fidCap, '%s\t%s\n', stem, strrep(caption, sprintf('\t'), ' '));
end

function phase7HatchRect(ax, x0, x1, y0, y1, kind, color, logy)
    % Opaque hatching inside a rectangle, drawn as line segments.
    if strcmp(kind, 'none'), return; end
    if logy, v0 = log10(y0); v1 = log10(y1); else, v0 = y0; v1 = y1; end
    toY = @(v) v; if logy, toY = @(v) 10 .^ v; end
    L = linspace(0, 1, 7); L = L(2:end - 1);
    seg = zeros(0, 4);   % normalised [s0 t0 s1 t1]
    if any(strcmp(kind, {'fwd', 'cross'}))
        for c = linspace(-0.8, 0.8, 7)
            s = [max(0, c), min(1, 1 + c)]; seg(end + 1, :) = [s(1), s(1) - c, s(2), s(2) - c]; %#ok<AGROW>
        end
    end
    if any(strcmp(kind, {'bwd', 'cross'}))
        for c = linspace(0.2, 1.8, 7)
            s = [max(0, c - 1), min(1, c)]; seg(end + 1, :) = [s(1), c - s(1), s(2), c - s(2)]; %#ok<AGROW>
        end
    end
    if any(strcmp(kind, {'horiz', 'plus'}))
        for t = L, seg(end + 1, :) = [0, t, 1, t]; end %#ok<AGROW>
    end
    if any(strcmp(kind, {'vert', 'plus'}))
        for s = L, seg(end + 1, :) = [s, 0, s, 1]; end %#ok<AGROW>
    end
    for k = 1:size(seg, 1)
        xs = x0 + seg(k, [1 3]) * (x1 - x0);
        vs = v0 + seg(k, [2 4]) * (v1 - v0);
        line(ax, xs, toY(vs), 'Color', color, 'LineWidth', 0.5);
    end
    if strcmp(kind, 'dots')
        [S, Tt] = meshgrid(L, L);
        plot(ax, x0 + S(:) * (x1 - x0), toY(v0 + Tt(:) * (v1 - v0)), '.', 'Color', color, 'MarkerSize', 4);
    end
end

function phase7Bars(ax, V, LO, insts, methods, logy, annotateMissing)
    % Grouped bars: rows = instances, columns = methods. LO (optional, same
    % size) gives the lower stacked segment (B1: UL evaluations).
    nI = numel(insts); nM = numel(methods);
    w = 0.84 / nM; base = 0; if logy, base = 1; end
    for k = 1:nM
        st = phase7Style(methods{k});
        light = lighten(st.color, 0.15);
        if strcmp(methods{k}, 'M1'), edge = [0.04 0.04 0.04]; else, edge = st.color; end
        for i = 1:nI
            x0 = i - 0.42 + w * (k - 1) + 0.05 * w; x1 = x0 + 0.9 * w;
            v = V(i, k);
            if ~isfinite(v)
                if annotateMissing
                    text(ax, (x0 + x1) / 2, base + 0.8, 'target not reached', 'Rotation', 90, ...
                        'FontSize', 5, 'Color', [0.32 0.32 0.31], 'HorizontalAlignment', 'left', ...
                        'VerticalAlignment', 'middle');
                end
                continue;
            end
            if ~isempty(LO)
                lo = LO(i, k);
                patch(ax, [x0 x1 x1 x0], [base base lo lo], st.color, 'EdgeColor', st.color, 'LineWidth', 0.4);
                patch(ax, [x0 x1 x1 x0], [lo lo v v], lighten(st.color, 0.6), 'EdgeColor', edge, 'LineWidth', 0.4);
                phase7HatchRect(ax, x0, x1, lo, v, st.hatch, st.color, logy);
            else
                patch(ax, [x0 x1 x1 x0], [base base v v], light, 'EdgeColor', edge, 'LineWidth', 0.5);
                phase7HatchRect(ax, x0, x1, base, v, st.hatch, edge, logy);
            end
        end
    end
    set(ax, 'XTick', 1:nI, 'XTickLabel', cellfun(@(s) strtok(s, '_'), insts, 'UniformOutput', false), ...
        'XLim', [0.45, nI + 0.55], 'FontSize', 7);
    if logy, set(ax, 'YScale', 'log'); end
    hs = gobjects(1, nM); labs = cell(1, nM);
    for k = 1:nM
        st = phase7Style(methods{k});
        hs(k) = patch(ax, NaN(1, 4), NaN(1, 4), lighten(st.color, 0.15), 'EdgeColor', st.color);
        labs{k} = sprintf('%s [%s]', phase7Label(methods{k}), st.hatch);
    end
    addLegend(ax, hs, labs, 'northoutside', 6);
end

% ---------------------------------------------------------------- orchestrator
function phase7Figures(R, suite, cfg, figDir, quick)
    if ~exist(figDir, 'dir'), mkdir(figDir); end
    o = phase7Options(quick);
    fidCap = fopen(fullfile(figDir, 'phase7_captions.tsv'), 'w');
    fidFid = fopen(fullfile(figDir, 'phase7_fidelity.csv'), 'w');
    fprintf(fidFid, 'part,instance,run,stage,region,n,rmse,max_abs,coverage_2sd,mean_sd,reproduces,max_rel_dev\n');
    have = @(id) any(strcmp(R.instanceIds, id)) && any(strcmp(R.methodIds, 'M1'));
    if have('TP3')
        phase7PartA_TP3(R, suite, cfg, figDir, o, fidCap, fidFid);
    else
        fprintf('    Part A (TP3) skipped: TP3 is not in this sweep.\n');
    end
    for id = {'TP3', 'TP4', 'TP5', 'TP7', 'TP8'}
        if have(id{1}), phase7PartA3(R, suite, cfg, figDir, o, id{1}, fidCap, fidFid); end
    end
    a5 = [arrayfun(@(k) sprintf('TP%d', k), 1:8, 'UniformOutput', false), ...
          arrayfun(@(k) sprintf('SMD%d_p3q3', k), 1:12, 'UniformOutput', false)];
    for id = a5
        if have(id{1}), phase7PartA5(R, suite, cfg, figDir, o, id{1}, fidCap, fidFid); end
    end
    if have('SMD6_p3q3'), phase7PartA6(R, suite, cfg, figDir, o, fidCap, fidFid); end
    phase7PartB(R, figDir, o, fidCap);
    fclose(fidCap); fclose(fidFid);
end

% ---------------------------------------------------------------- Part A1, A2, A4 (TP3)
function phase7PartA_TP3(R, suite, cfg, figDir, o, fidCap, fidFid)
    inst = phase7Inst(suite, 'TP3');
    g = linspace(inst.xb(1, 1), inst.xb(1, 2), o.nGrid);
    [X1, X2] = meshgrid(g, g);
    TH = zeros(size(X1)); FS = TH; Y1 = TH; Y2 = TH;
    for a = 1:numel(X1)
        T = phase7Truth(inst, [X1(a), X2(a)], o);
        TH(a) = T.theta; FS(a) = T.fstar; Y1(a) = T.y(1); Y2(a) = T.y(2);
    end
    % bilevel optimum, computed (not copied): a 301 x 301 grid over the UL box,
    % then a 101 x 101 grid of step 0.002 around the best UL-feasible grid point
    best = Inf; xs = [NaN NaN];
    for pass = 1:2
        if pass == 1
            g1 = linspace(inst.xb(1, 1), inst.xb(1, 2), 301); g2 = linspace(inst.xb(2, 1), inst.xb(2, 2), 301);
        else
            g1 = min(max(xs(1) + (-50:50) * 0.002, inst.xb(1, 1)), inst.xb(1, 2));
            g2 = min(max(xs(2) + (-50:50) * 0.002, inst.xb(2, 1)), inst.xb(2, 2));
        end
        [Q1, Q2] = meshgrid(g1, g2);
        for a = 1:numel(Q1)
            x = [Q1(a), Q2(a)]; y = phase7Ystar(inst, x, o);
            if all(evalG(inst, x, y) <= 1e-12)
                v = inst.F(x, y);
                if v < best, best = v; xs = x; end
            end
        end
    end
    ys = phase7Ystar(inst, xs, o); fsStar = inst.f(xs, ys);
    gl = linspace(inst.xb(1, 1), inst.xb(1, 2), 200);
    % A1: f* and theta wireframes
    for which = 1:2
        [fig, ax] = phase7Fig3d();
        if which == 1, Z = FS; zs = fsStar; zl = 'f^*(x)'; stem = 'A1a_tp3_llvalue_truth';
        else, Z = TH; zs = best; zl = '\theta(x)'; stem = 'A1b_tp3_theta_truth'; end
        phase7Wire(ax, X1, X2, Z, [0.32 0.32 0.31], 0.45, '-', 1);
        z0 = min(Z(:));
        h1 = plot3(ax, gl, (30 - gl) / 2, z0 * ones(size(gl)), '-', 'Color', [0.165 0.471 0.839], 'LineWidth', 1);
        plot3(ax, gl, 25 - gl, z0 * ones(size(gl)), '--', 'Color', [0.165 0.471 0.839], 'LineWidth', 1);
        h2 = plot3(ax, xs(1), xs(2), zs, 'o', 'MarkerSize', 7, 'MarkerFaceColor', 'w', ...
            'MarkerEdgeColor', [0.89 0.29 0.28], 'LineWidth', 1.6);
        zlabel(ax, zl); xlim(ax, inst.xb(1, :)); ylim(ax, inst.xb(2, :));
        addLegend(ax, [h2 h1], {sprintf('bilevel optimum x* = (%.4g, %.4g)', xs), 'UL constraint boundaries'}, 'northwest', 7);
        cap = sprintf(['TP3 ground truth on a %d x %d grid over the UL box. The LL problem is a strictly ' ...
            'convex QP over a box, so y*(x) = min(x, 10) is single-valued (Assumption A2); exact projection. ' ...
            'Optimum x* = (%.4g, %.4g), y* = (%.4g, %.4g), theta = %.6g, f* = %.6g (grid search over the ' ...
            'UL-feasible set, refined locally). Blue: UL constraint boundaries. Ground truth only.'], ...
            o.nGrid, o.nGrid, xs, ys, best, fsStar);
        phase7Save(fig, figDir, stem, cap, fidCap);
    end
    % A4: y* components
    for k = 1:2
        [fig, ax] = phase7Fig3d();
        if k == 1, Z = Y1; else, Z = Y2; end
        phase7Wire(ax, X1, X2, Z, [0.32 0.32 0.31], 0.45, '-', 1);
        zlabel(ax, sprintf('y^*_%d(x)', k));
        cap = sprintf(['TP3, true follower response y*_%d(x) = min(x_%d, 10), from the exact solution of the ' ...
            'LL quadratic programme. SA-APPA does not model y*(x); no approximation is drawn.'], k, k);
        phase7Save(fig, figDir, sprintf('A4_tp3_ystar%d', k), cap, fidCap);
    end
    % A2: the GP SA-APPA held, lower-median run
    [runId, idx] = phase7MedianRun(R, 'M1', 'TP3');
    [snaps, chk] = phase7Rerun(R, inst, 'M1', runId, idx, cfg);
    fprintf('    TP3 run %d re-run: max relative deviation %.2g (reproduces = %d)\n', runId, chk.maxRelDev, chk.reproduces);
    if ~chk.reproduces
        warning('phase7: TP3 run %d does not reproduce; Part A2 skipped.', runId);
        return;
    end
    Xh = lhsSample(newStream(o.heldSeed), inst.xb, o.nHeldTP3);
    th = zeros(size(Xh, 1), 1);
    for a = 1:size(Xh, 1), th(a) = inst.F(Xh(a, :), phase7Ystar(inst, Xh(a, :), o)); end
    sN = newStream(o.heldSeed); Xn = zeros(0, 2);
    while size(Xn, 1) < 500
        z = xs + o.rNear * (2 * rand(sN, 1, 2) - 1);
        if norm(z - xs) <= o.rNear && all(z >= inst.xb(:, 1)') && all(z <= inst.xb(:, 2)'), Xn(end + 1, :) = z; end %#ok<AGROW>
    end
    tn = zeros(size(Xn, 1), 1);
    for a = 1:size(Xn, 1), tn(a) = inst.F(Xn(a, :), phase7Ystar(inst, Xn(a, :), o)); end
    feasH = (Xh(:, 1) + 2 * Xh(:, 2) >= 30) & (Xh(:, 1) + Xh(:, 2) <= 25);
    stages = {'init', 'final'}; letters = {'abc', 'def'};
    for si = 1:2
        sn = snaps.(stages{si});
        GXX = [X1(:), X2(:)];
        [mu, sd] = surrogatePredict(sn.theta, GXX); MU = reshape(mu, size(X1)); SD = reshape(sd, size(X1));
        regions = {'whole_box', Xh, th; 'near_optimum', Xn, tn; 'feasible_region', Xh(feasH, :), th(feasH)};
        Mx = struct();
        for r = 1:3
            M = phase7Metrics(sn.theta, regions{r, 2}, regions{r, 3});
            Mx.(regions{r, 1}) = M;
            fprintf(fidFid, 'A2,TP3,%d,%s,%s,%d,%.6g,%.6g,%.4f,%.4g,%d,%.3g\n', runId, stages{si}, regions{r, 1}, ...
                M.n, M.rmse, M.maxAbs, M.coverage, M.meanSd, chk.reproduces, chk.maxRelDev);
        end
        ok = all(sn.G <= 1e-6, 2); Fk = sn.F; Fk(~ok) = Inf; [incF, kk] = min(Fk); inc = sn.X(kk, :);
        common = sprintf(['TP3, SA-APPA (M1), run %d (lower-median of the M1 runs, rule fixed in advance); the GP ' ...
            'SA-APPA held %s (%d training points, lengthscale %.3g by the median heuristic, signal variance 1, ' ...
            'outputs not rescaled; nothing tuned).'], runId, ...
            ternary(si == 1, 'after the initial design', 'at the end of the run'), numel(sn.F), sn.theta.lengthscale);
        % (i) truth vs mean
        [fig, ax] = phase7Fig3d();
        h1 = phase7Wire(ax, X1, X2, TH, [0.32 0.32 0.31], 0.45, '-', 2);
        h2 = phase7Wire(ax, X1, X2, MU, [0.165 0.471 0.839], 0.55, '-', 2);
        h3 = plot3(ax, sn.X(:, 1), sn.X(:, 2), sn.F, '.', 'Color', [0.04 0.04 0.04], 'MarkerSize', 9);
        h4 = plot3(ax, xs(1), xs(2), best, 'o', 'MarkerSize', 7, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', [0.89 0.29 0.28], 'LineWidth', 1.6);
        h5 = plot3(ax, inc(1), inc(2), incF, 'x', 'MarkerSize', 9, 'Color', [0.89 0.29 0.28], 'LineWidth', 2);
        zlabel(ax, '\theta');
        addLegend(ax, [h1 h2 h3 h4 h5], {'true \theta', 'GP mean \mu_\theta', 'training points', 'optimum x*', 'SA-APPA incumbent'}, 'northwest', 6);
        cap = sprintf('%s True theta (grey), GP mean (blue). Held-out RMSE %.3g over the box, %.3g within %.3g of x*.', ...
            common, Mx.whole_box.rmse, Mx.near_optimum.rmse, o.rNear);
        phase7Save(fig, figDir, sprintf('A2%s_tp3_theta_gp_mean_%s', letters{si}(1), stages{si}), cap, fidCap);
        % (ii) mean +/- 2 sd
        [fig, ax] = phase7Fig3d();
        h1 = phase7Wire(ax, X1, X2, TH, [0.32 0.32 0.31], 0.45, '-', 2);
        h2 = phase7Wire(ax, X1, X2, MU + 2 * SD, [0.922 0.408 0.204], 0.5, '-', 2);
        h3 = phase7Wire(ax, X1, X2, MU - 2 * SD, [0.290 0.227 0.655], 0.5, '--', 2);
        zlabel(ax, '\theta');
        addLegend(ax, [h1 h2 h3], {'true \theta', '\mu_\theta + 2\sigma_\theta', '\mu_\theta - 2\sigma_\theta'}, 'northwest', 6);
        cap = sprintf(['%s Surfaces mu +/- 2 sigma (multiplier 2). Coverage of the +/- 2 sigma interval: %.1f%% over ' ...
            'the box, %.1f%% near x* (nominal 95%%); mean sigma %.3g.'], common, 100 * Mx.whole_box.coverage, ...
            100 * Mx.near_optimum.coverage, Mx.whole_box.meanSd);
        phase7Save(fig, figDir, sprintf('A2%s_tp3_theta_gp_band_%s', letters{si}(2), stages{si}), cap, fidCap);
        % (iii) error map
        fig = figure('Visible', 'off', 'Color', 'w', 'Units', 'inches', 'Position', [1 1 4.0 3.4]);
        ax = axes('Parent', fig); hold(ax, 'on'); box(ax, 'on');
        E = abs(MU - TH);
        contourf(ax, X1, X2, E, 10, 'LineColor', 'none');
        colormap(ax, phase7Seq(64)); cb = colorbar(ax); ylabel(cb, '|\mu_\theta - \theta|');
        plot(ax, gl, (30 - gl) / 2, '-k', 'LineWidth', 0.8); plot(ax, gl, 25 - gl, '--k', 'LineWidth', 0.8);
        plot(ax, [10 10], inst.xb(2, :), ':', 'Color', [0.54 0.54 0.52]); plot(ax, inst.xb(1, :), [10 10], ':', 'Color', [0.54 0.54 0.52]);
        plot(ax, sn.X(:, 1), sn.X(:, 2), '.k', 'MarkerSize', 9);
        plot(ax, xs(1), xs(2), 'o', 'MarkerSize', 7, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', [0.89 0.29 0.28], 'LineWidth', 1.5);
        xlim(ax, inst.xb(1, :)); ylim(ax, inst.xb(2, :)); xlabel(ax, 'x_1'); ylabel(ax, 'x_2');
        [~, im] = max(E(:));
        cap = sprintf(['%s Absolute error of the GP mean; largest at (%.3g, %.3g). Dots: training points; circle: ' ...
            'optimum; black: UL constraint boundaries; dotted: LL kink lines x_i = 10.'], common, X1(im), X2(im));
        phase7Save(fig, figDir, sprintf('A2%s_tp3_theta_gp_error_%s', letters{si}(3), stages{si}), cap, fidCap);
    end
end

function out = ternary(c, a, b)
    if c, out = a; else, out = b; end
end

% ---------------------------------------------------------------- Part A3 (UL constraint maps)
function phase7PartA3(R, suite, cfg, figDir, o, id, fidCap, fidFid)
    inst = phase7Inst(suite, id);
    [runId, idx] = phase7MedianRun(R, 'M1', id);
    [snaps, chk] = phase7Rerun(R, inst, 'M1', runId, idx, cfg);
    if ~chk.reproduces
        warning('phase7: %s run %d does not reproduce; A3 skipped.', id, runId); return;
    end
    sn = snaps.final;
    if isempty(sn.phi), return; end
    g1 = linspace(inst.xb(1, 1), inst.xb(1, 2), o.nGrid); g2 = linspace(inst.xb(2, 1), inst.xb(2, 2), o.nGrid);
    [X1, X2] = meshgrid(g1, g2); GXX = [X1(:), X2(:)];
    m = numel(sn.phi); PHI = zeros(numel(X1), m);
    for a = 1:numel(X1)
        x = GXX(a, :);
        if strcmp(id, 'TP8'), y = phase7Ystar(inst, x, o); else, y = mean(inst.yb, 2)'; end   % other TPs: phi depends on x only
        PHI(a, :) = evalG(inst, x, y);
    end
    ok = all(sn.G <= 1e-6, 2);
    for j = 1:m
        [mu, sd] = surrogatePredict(sn.phi{j}, GXX);
        e = mu - PHI(:, j);
        fprintf(fidFid, 'A3,%s,%d,final,phi%d,%d,%.6g,%.6g,%.4f,%.4g,%d,%.3g\n', id, runId, j, numel(e), ...
            sqrt(mean(e .^ 2)), max(abs(e)), mean(abs(e) <= 2 * sd), mean(sd), chk.reproduces, chk.maxRelDev);
        [fig, ax] = phase7Fig3d();
        h1 = phase7Wire(ax, X1, X2, reshape(PHI(:, j), size(X1)), [0.32 0.32 0.31], 0.45, '-', 2);
        h2 = phase7Wire(ax, X1, X2, reshape(mu, size(X1)), [0.165 0.471 0.839], 0.55, '-', 2);
        h3 = plot3(ax, sn.X(:, 1), sn.X(:, 2), sn.G(:, j), '.k', 'MarkerSize', 9);
        hs = [h1 h2 h3]; labs = {sprintf('true \\phi_%d', j), sprintf('GP mean of \\phi_%d', j), 'training points'};
        if any(ok)
            Fk = sn.F; Fk(~ok) = Inf; [~, kk] = min(Fk);
            hs(end + 1) = plot3(ax, sn.X(kk, 1), sn.X(kk, 2), 0, 'x', 'MarkerSize', 9, 'Color', [0.89 0.29 0.28], 'LineWidth', 2); %#ok<AGROW>
            labs{end + 1} = 'SA-APPA incumbent (at \phi = 0)'; %#ok<AGROW>
        end
        zlabel(ax, sprintf('\\phi_%d', j));
        addLegend(ax, hs, labs, 'northwest', 6);
        cap = sprintf(['%s, UL constraint phi_%d: true map (grey) and the GP SA-APPA held at the end of run %d ' ...
            '(lower-median run). RMSE %.3g, +/- 2 sigma coverage %.0f%%, sign agreement %.1f%%.'], id, j, runId, ...
            sqrt(mean(e .^ 2)), 100 * mean(abs(e) <= 2 * sd), 100 * mean(sign(mu) == sign(PHI(:, j))));
        phase7Save(fig, figDir, sprintf('A3_%s_phi%d', lower(id), j), cap, fidCap);
    end
end

% ---------------------------------------------------------------- Part A5 (held-out fidelity)
function phase7PartA5(R, suite, cfg, figDir, o, id, fidCap, fidFid)
    inst = phase7Inst(suite, id);
    [runId, idx] = phase7MedianRun(R, 'M1', id);
    [snaps, chk] = phase7Rerun(R, inst, 'M1', runId, idx, cfg);
    if ~chk.reproduces
        warning('phase7: %s run %d does not reproduce; A5 skipped.', id, runId); return;
    end
    if strncmp(id, 'TP', 2), n = o.nHeldTP; else, n = o.nHeldSMD; end
    Xh = lhsSample(newStream(o.heldSeed), inst.xb, n);
    th = NaN(n, 1); keep = false(n, 1);
    for a = 1:n
        T = phase7Truth(inst, Xh(a, :), o);
        th(a) = T.theta; keep(a) = T.llFeasible;
    end
    [mu, sd] = surrogatePredict(snaps.final.theta, Xh(keep, :));
    M = phase7Metrics(snaps.final.theta, Xh(keep, :), th(keep));
    fprintf(fidFid, 'A5,%s,%d,final,heldout,%d,%.6g,%.6g,%.4f,%.4g,%d,%.3g\n', id, runId, M.n, M.rmse, M.maxAbs, ...
        M.coverage, M.meanSd, chk.reproduces, chk.maxRelDev);
    fig = figure('Visible', 'off', 'Color', 'w', 'Units', 'inches', 'Position', [1 1 3.4 3.2]);
    ax = axes('Parent', fig); hold(ax, 'on'); box(ax, 'on');
    t = th(keep);
    hEb = errorbar(ax, t, mu, 2 * sd, 'o');
    set(hEb, 'MarkerSize', 3, 'Color', [0.165 0.471 0.839], 'LineWidth', 0.5);
    try, set(hEb, 'CapSize', 0); catch, end
    lims = [min([t; mu]), max([t; mu])];
    plot(ax, lims, lims, '--', 'Color', [0.32 0.32 0.31]);
    xlabel(ax, 'true \theta(x)'); ylabel(ax, 'GP mean \pm 2\sigma'); title(ax, strrep(id, '_', ' '), 'FontSize', 8);
    cap = sprintf(['%s: GP mean of theta against the true theta on %d held-out LHS points (%d dropped: no LL-feasible ' ...
        'response), +/- 2 sigma bars, GP of run %d at the end of the run. RMSE %.3g, coverage %.0f%% (nominal 95%%). ' ...
        'The mean predictive standard deviation is %.2g, so most bars are narrower than the markers.'], ...
        id, M.n, sum(~keep), runId, M.rmse, 100 * M.coverage, M.meanSd);
    phase7Save(fig, figDir, sprintf('A5_%s', lower(id)), cap, fidCap);
end

% ---------------------------------------------------------------- Part A6 (SMD6, non-unique LL response)
function phase7PartA6(R, suite, cfg, figDir, o, fidCap, fidFid)
    % SMD6 has a continuum of LL optima (t1 = t2 for the s = 2 tail variables) while the UL
    % objective adds +sum(t.^2), so the optimistic response has t = 0 and
    %   theta_opt(x) = sum(xu1.^2) + sum(xu2.^2) - sum((xu2 - clip(xu2, -pi/2, pi/2)).^2).
    % The nested DE solve returns an arbitrary point on the optimal line instead. This part
    % compares SA-APPA's training targets with theta_opt and fits a diagnostic GP (same rule,
    % same inputs) to theta_opt; the diagnostic GP is not a run of any configuration.
    id = 'SMD6_p3q3';
    inst = phase7Inst(suite, id);
    [runId, idx] = phase7MedianRun(R, 'M1', id);
    [snaps, chk] = phase7Rerun(R, inst, 'M1', runId, idx, cfg);
    if ~chk.reproduces
        warning('phase7: %s run %d does not reproduce; A6 skipped.', id, runId); return;
    end
    pU = size(inst.xb, 1) - 2;
    lo = -pi / 2 + 1e-6; hi = pi / 2 - 1e-6;
    thOpt = @(X) sum(X(:, 1:pU) .^ 2, 2) + sum(X(:, pU + 1:end) .^ 2, 2) ...
        - sum((X(:, pU + 1:end) - min(max(X(:, pU + 1:end), lo), hi)) .^ 2, 2);
    Xt = snaps.final.X; Ft = snaps.final.F(:); Fo = thOpt(Xt);
    gap = Ft - Fo;
    Xh = lhsSample(newStream(o.heldSeed), inst.xb, o.nHeldSMD);
    to = thOpt(Xh);
    [mu, ~] = surrogatePredict(snaps.final.theta, Xh);
    cf = gpFit(Xt, Fo);
    [muCf, ~] = gpPredict(cf, Xh);
    rmse = sqrt(mean((mu - to) .^ 2)); rmseCf = sqrt(mean((muCf - to) .^ 2));
    fprintf(fidFid, 'A6,%s,%d,final,heldout_optimistic,%d,%.6g,%.6g,,,%d,%.3g\n', id, runId, numel(to), rmse, ...
        max(abs(mu - to)), chk.reproduces, chk.maxRelDev);
    fprintf(fidFid, 'A6,%s,%d,diagnostic,heldout_optimistic,%d,%.6g,%.6g,,,%d,%.3g\n', id, runId, numel(to), rmseCf, ...
        max(abs(muCf - to)), chk.reproduces, chk.maxRelDev);
    blue = [0.165 0.471 0.839]; grey = [0.32 0.32 0.31];

    fig = figure('Visible', 'off', 'Color', 'w', 'Units', 'inches', 'Position', [1 1 3.4 3.2]);
    ax = axes('Parent', fig); hold(ax, 'on'); box(ax, 'on');
    plot(ax, Fo, Ft, 'o', 'MarkerSize', 3, 'MarkerFaceColor', blue, 'MarkerEdgeColor', blue);
    lims = [min([Fo; Ft]), max([Fo; Ft])];
    plot(ax, lims, lims, '--', 'Color', grey);
    xlabel(ax, 'optimistic \theta(x)'); ylabel(ax, 'value returned by the nested LL solve');
    title(ax, 'SMD6 (3, 3), training points', 'FontSize', 8);
    cap = sprintf(['SMD6 at (3, 3): the %d UL values SA-APPA trained its GP on (run %d, lower-median run) against ' ...
        'the optimistic theta at the same points. Gap median %.1f, maximum %.1f, minimum %.3g (non-negative).'], ...
        numel(Ft), runId, median(gap), max(gap), min(gap));
    phase7Save(fig, figDir, 'A6a_smd6_training_targets', cap, fidCap);

    fig = figure('Visible', 'off', 'Color', 'w', 'Units', 'inches', 'Position', [1 1 3.4 3.2]);
    ax = axes('Parent', fig); hold(ax, 'on'); box(ax, 'on');
    h1 = plot(ax, to, mu, 's', 'MarkerSize', 3, 'MarkerFaceColor', [0.62 0.62 0.61], 'MarkerEdgeColor', [0.62 0.62 0.61]);
    h2 = plot(ax, to, muCf, 'o', 'MarkerSize', 3, 'MarkerFaceColor', blue, 'MarkerEdgeColor', blue);
    lims = [min([to; mu; muCf]), max([to; mu; muCf])];
    plot(ax, lims, lims, '--', 'Color', grey);
    xlabel(ax, 'optimistic \theta(x)'); ylabel(ax, 'GP mean \mu_\theta(x)');
    title(ax, 'SMD6 (3, 3), held-out points', 'FontSize', 8);
    try
        legend(ax, [h1 h2], {'SA-APPA''s GP (targets as observed)', 'same GP rule, optimistic targets'}, ...
            'Location', 'northwest', 'FontSize', 6.5);
    catch
    end
    cap = sprintf(['SMD6 at (3, 3), %d held-out points: GP mean against the optimistic theta. Grey: the GP SA-APPA ' ...
        'held at the end of run %d (RMSE %.0f). Blue: a diagnostic GP with the same kernel, hyperparameter rule and ' ...
        'training inputs, trained on the optimistic theta (RMSE %.1f). Not an SA-APPA run.'], numel(to), runId, rmse, rmseCf);
    phase7Save(fig, figDir, 'A6b_smd6_consistent_targets', cap, fidCap);
end

% ---------------------------------------------------------------- Part B
function phase7PartB(R, figDir, o, fidCap)
    groups = struct('key', {'ablA', 'ablB', 'ext', 'all'}, ...
        'methods', {{'M1', 'M2', 'M3', 'M4', 'M5', 'M6'}, {'M1', 'M7', 'M8', 'M9'}, ...
                    {'M1', 'M10', 'M11', 'M12', 'M13'}, R.methodIds});
    sets = struct('key', {'tp', 'smd_s', 'smd_m', 'smd_l'}, ...
        'insts', {arrayfun(@(k) sprintf('TP%d', k), 1:9, 'UniformOutput', false), ...
                  arrayfun(@(k) sprintf('SMD%d_p3q3', k), 1:12, 'UniformOutput', false), ...
                  arrayfun(@(k) sprintf('SMD%d_p10q10', k), 1:12, 'UniformOutput', false), ...
                  arrayfun(@(k) sprintf('SMD%d_p30q30', k), 1:12, 'UniformOutput', false)});
    nI = numel(R.instanceIds);
    % reference values: best final feasible value over all configurations and
    % runs; on SMD1-SMD9 (UL optimum F* = 0) the reference is 0
    ref = NaN(1, nI);
    for i = 1:nI
        sel = R.ii == i & R.feas;
        if any(sel), ref(i) = min(R.best(sel)); end
        tok = regexp(R.instanceIds{i}, '^SMD(\d+)_', 'tokens', 'once');
        if ~isempty(tok) && str2double(tok{1}) <= 9, ref(i) = 0; end
    end
    % B2: evaluations to target, lower median (k-th of n, k = ceil(n/2)); undefined if fewer than k reach
    for ti = 1:numel(o.targets)
        tol = o.targets(ti);
        fidS = fopen(fullfile(figDir, sprintf('phase7_B2_success_t%d.csv', ti)), 'w');
        fprintf(fidS, 'instance_id,method_id,median_nested_solves,n_reached,n_runs\n');
        MED = NaN(nI, numel(R.methodIds));
        for i = 1:nI
            for m = 1:numel(R.methodIds)
                sel = find(R.ii == i & R.mi == m);
                hit = Inf(numel(sel), 1);
                for q = 1:numel(sel)
                    rows = R.rows{sel(q)};
                    if isempty(rows), continue; end
                    h = find(abs(rows(:, 3) - ref(i)) <= tol, 1);
                    if ~isempty(h), hit(q) = rows(h, 1); end
                end
                hs = sort(hit); k = ceil(numel(hs) / 2);
                if ~isempty(hs) && isfinite(hs(k)), MED(i, m) = hs(k); end
                fprintf(fidS, '%s,%s,%s,%d,%d\n', R.instanceIds{i}, R.methodIds{m}, num2str(MED(i, m)), ...
                    sum(isfinite(hit)), numel(sel));
            end
        end
        fclose(fidS);
        for s = sets
            if ti == 2 && strcmp(s.key, 'tp'), continue; end
            ins = s.insts(ismember(s.insts, R.instanceIds));
            if isempty(ins), continue; end
            iIdx = cellfun(@(z) find(strcmp(R.instanceIds, z)), ins);
            for gr = groups
                ms = gr.methods(ismember(gr.methods, R.methodIds));
                mIdx = cellfun(@(z) find(strcmp(R.methodIds, z)), ms);
                fig = figure('Visible', 'off', 'Color', 'w', 'Units', 'inches', ...
                    'Position', [1 1 max(5.6, 0.16 * numel(ins) * numel(ms) + 1.2) 3.4]);
                ax = axes('Parent', fig); hold(ax, 'on'); box(ax, 'off'); grid(ax, 'on');
                try, set(ax, 'XGrid', 'off', 'GridAlpha', 1, 'GridColor', [0.89 0.89 0.87]); catch, end
                phase7Bars(ax, MED(iIdx, mIdx), [], ins, ms, false, true);
                ylim(ax, [0 62]); ylabel(ax, 'median nested solves to target');
                cap = sprintf(['Evaluations to target %g (absolute) of the reference value, set %s, group %s. Median over ' ...
                    'the %d runs (not 31) of the nested solves after which the best feasible UL objective first lies ' ...
                    'within the target; reference = best value found by any configuration, except 0 on SMD1-SMD9. ' ...
                    'Censored runs never reach the target; with fewer than half reaching it, no bar is drawn.'], ...
                    tol, s.key, gr.key, numel(unique(R.run)));
                phase7Save(fig, figDir, sprintf('B2_t%d_%s_%s', ti, s.key, gr.key), cap, fidCap);
            end
        end
    end
    % B1: budget composition, all runs (LL counts from llEvalCounter)
    NUL = NaN(nI, numel(R.methodIds)); NLL = NUL;
    fidB = fopen(fullfile(figDir, 'phase7_B1_budget.csv'), 'w');
    fprintf(fidB, 'instance_id,method_id,median_ul,median_ll,n_runs\n');
    for i = 1:nI
        for m = 1:numel(R.methodIds)
            sel = find(R.ii == i & R.mi == m & ~isnan(R.nll));
            if isempty(sel), continue; end
            nul = cellfun(@(r) size(r, 1), R.rows(sel));
            NUL(i, m) = median(nul); NLL(i, m) = median(R.nll(sel));
            fprintf(fidB, '%s,%s,%g,%g,%d\n', R.instanceIds{i}, R.methodIds{m}, NUL(i, m), NLL(i, m), numel(sel));
        end
    end
    fclose(fidB);
    for s = sets
        ins = s.insts(ismember(s.insts, R.instanceIds));
        if isempty(ins), continue; end
        iIdx = cellfun(@(z) find(strcmp(R.instanceIds, z)), ins);
        for gr = groups
            ms = gr.methods(ismember(gr.methods, R.methodIds));
            mIdx = cellfun(@(z) find(strcmp(R.methodIds, z)), ms);
            fig = figure('Visible', 'off', 'Color', 'w', 'Units', 'inches', ...
                'Position', [1 1 max(5.6, 0.16 * numel(ins) * numel(ms) + 1.2) 3.4]);
            ax = axes('Parent', fig); hold(ax, 'on'); box(ax, 'off'); grid(ax, 'on');
            try, set(ax, 'XGrid', 'off', 'GridAlpha', 1, 'GridColor', [0.89 0.89 0.87]); catch, end
            phase7Bars(ax, NUL(iIdx, mIdx) + NLL(iIdx, mIdx), NUL(iIdx, mIdx), ins, ms, true, false);
            ylabel(ax, 'median evaluations (log scale)');
            cap = sprintf(['Budget composition, set %s, group %s: median over all runs of UL evaluations (solid lower ' ...
                'segment, full nested solves) plus LL objective evaluations inside them (hatched upper segment), counted ' ...
                'by llEvalCounter(). Every LL solve is a nested heuristic DE solve (population %d n_ll, at most %d ' ...
                'generations, polish); none is exact. Log axis: segment heights are not proportional to counts.'], ...
                s.key, gr.key, 6, 20);
            phase7Save(fig, figDir, sprintf('B1_%s_%s', s.key, gr.key), cap, fidCap);
        end
    end
end
