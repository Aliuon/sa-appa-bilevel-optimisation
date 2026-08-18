function seed_rng(seed)
% SEED_RNG  Seeds the random number generator, working across both
% MATLAB (where rng() is preferred and rand('seed',...) is deprecated)
% and Octave (which does not have rng() before fairly recent versions,
% but does support rand('seed', ...) and rand('state', ...)). Tries the
% modern MATLAB API first and falls back to the legacy call so this one
% function works unmodified in either environment -- untested in both,
% per this package's standing caveat (see README_MATLAB.md).
    try
        rng(seed); %#ok<RNG>
    catch
        try
            rand('seed', seed); %#ok<RAND>
        catch
            rand('state', seed); %#ok<RAND>
        end
    end
end
