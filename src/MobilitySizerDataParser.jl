module MobilitySizerDataParser

# +
# Parser and inversion for DMPS (differential mobility particle sizer) files.
#
# File format (repeated blocks):
#   '<start timestamp>' '<stop timestamp>'      mm-dd-yyyy HH:MM:SS
#   <15 whitespace separated header values>      see parse_header
#   <voltage>  <number concentration>            one row per voltage step
#
# Author: Jens Top
# 2026-10-07 fixes: mobility grid + stepped-DMA kernel + distribution moments
#-

using DataFrames
using Dates
using CSV
using Plots

using DifferentialMobilityAnalyzers
using DifferentialMobilityAnalyzers:
    DMAconfig,
    DifferentialMobilityAnalyzer,
    SizeDistribution,
    setupDMAgridded,
    vtoz,
    ztod,
    rinv2
using Interpolations: interpolate, extrapolate, Gridded, Linear

export Scan,
    DMPSResult,
    read_dmps,
    grid_from_voltages,
    invert_scan,
    distributions,
    moments,
    parse_file,
    process_file,
    banana_plot

# ---------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------

"""
    Scan

One voltage sweep of the DMPS.

- `t0`, `t1`: start and stop timestamps
- `Λ`: the [`DifferentialMobilityAnalyzers.DMAconfig`](@ref) built from the block header
- `V`: setpoint voltages [V], ascending
- `N`: raw number concentration response [cm⁻³], one value per voltage
- `Dp`: apparent +1 mobility diameter [nm] at each setpoint
"""
struct Scan
    t0::DateTime
    t1::DateTime
    Λ::DMAconfig
    V::Vector{Float64}
    N::Vector{Float64}
    Dp::Vector{Float64}
end

"""
    DMPSResult

Result of inverting one scan.

Length-`n` vectors are ordered by ascending `Dp`:
- `Dp`: mobility diameter bin midpoints [nm]
- `N`: inverted number concentration per bin [cm⁻³]
- `dN`: dN/dlog₁₀Dp [cm⁻³]
- `dV`: dV/dlog₁₀Dp [µm³ cm⁻³]
- `dm`: dm/dlog₁₀Dp [µg m⁻³]

Scalars:
- `Ntot`: total number concentration [cm⁻³]
- `Vtot`: total volume concentration [µm³ cm⁻³]
- `Mtot`: total mass concentration [µg m⁻³]
- `DgN`: number-weighted geometric mean diameter [nm]
- `DgM`: mass-weighted geometric mean diameter [nm]
"""
struct DMPSResult
    Dp::Vector{Float64}
    N::Vector{Float64}
    dN::Vector{Float64}
    dV::Vector{Float64}
    dm::Vector{Float64}
    Ntot::Float64
    Vtot::Float64
    Mtot::Float64
    DgN::Float64
    DgM::Float64
end

# ---------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------

const TS_REGEX = r"'([^']+)'\s+'([^']+)'"
const DATEFMT = "mm-dd-yyyy HH:MM:SS"
const NHEADER = 15

"""
    parse_header(vals, polarity = :+, m = 6, leff = 0.0)

Build a `DMAconfig` from the 15 header values of a block.

The header is assumed to be
```
1  sheath flow   [L min⁻¹]
2  aerosol flow  [L min⁻¹]
3  inner radius  [m]
4  outer radius  [m]
5  column length [m]
6  -
7  -
8  pressure      [hPa]
9  -
10 temperature   [°C]
11..15 -
```
Flows are converted L min⁻¹ → m³ s⁻¹, pressure hPa → Pa and temperature °C → K.
"""
function parse_header(vals, polarity = :+, m = 6, leff = 0.0)
    length(vals) == NHEADER ||
        error("expected $NHEADER header values, got $(length(vals))")

    qsh = parse(Float64, vals[1]) / 60.0 / 1000.0     # L/min -> m³/s
    qsa = parse(Float64, vals[2]) / 60.0 / 1000.0
    r₁ = parse(Float64, vals[3])
    r₂ = parse(Float64, vals[4])
    l = parse(Float64, vals[5])
    p = parse(Float64, vals[8]) * 100.0                # hPa -> Pa
    t = parse(Float64, vals[10]) + 273.15              # °C -> K

    return DMAconfig(t, p, qsa, qsh, r₁, r₂, l, leff, polarity, m, :cylindrical)
end

"""
    read_dmps(path; polarity = :+, m = 6, leff = 0.0)

Read a DMPS file and return a vector of [`Scan`](@ref)s.

Keyword arguments set the charge correction and diffusion-loss model:
- `polarity`: power supply polarity, `:+` or `:-`
- `m`: number of charge states used in the charge filter
- `leff`: effective length [m] for the DMA diffusion-loss correction.
  `leff = 0.0` disables the correction.
"""
function read_dmps(path; polarity = :+, m = 6, leff = 0.0)
    lines = readlines(path)
    scans = Scan[]
    i = 1

    while i <= length(lines)
        line = strip(lines[i])
        startswith(line, "'") || (i += 1; continue)

        mm = match(TS_REGEX, line)
        mm === nothing && error("invalid timestamp line at line $i")
        t0 = DateTime(mm.captures[1], DATEFMT)
        t1 = DateTime(mm.captures[2], DATEFMT)
        i += 1

        i > length(lines) && break
        vals = split(strip(lines[i]))
        Λ = parse_header(vals, polarity, m, leff)
        i += 1

        V = Float64[]
        N = Float64[]
        while i <= length(lines)
            row = strip(lines[i])
            startswith(row, "'") && break
            parts = split(row)
            length(parts) >= 2 || break
            push!(V, parse(Float64, parts[1]))
            push!(N, parse(Float64, parts[2]))
            i += 1
        end

        # a scan is only usable if it has at least three setpoints
        length(V) >= 3 || continue

        # sort ascending in voltage so that Dp is ascending
        o = sortperm(V)
        V = V[o]
        N = N[o]
        Dp = ztod.(Ref(Λ), 1, vtoz(Λ, V))

        push!(scans, Scan(t0, t1, Λ, V, N, Dp))
    end

    return scans
end

# ---------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------

"""
    mobility_edges(Z)

Construct mobility bin edges from a vector of bin midpoints `Z`.

Interior edges are the geometric means of adjacent midpoints, which is exact
for a geometrically spaced grid. The two outer edges are extrapolated so that
the first/last midpoint stays the geometric mean of its edges.
"""
function mobility_edges(Z::AbstractVector{<:Real})
    n = length(Z)
    n >= 2 || error("need at least two bin midpoints")
    Ze = Vector{Float64}(undef, n + 1)
    @inbounds for i in 2:n
        Ze[i] = sqrt(Z[i-1] * Z[i])
    end
    Ze[1] = Z[1]^2 / Ze[2]
    Ze[n+1] = Z[n]^2 / Ze[n]
    return Ze
end

"""
    grid_from_voltages(Λ, V)

Build the DMA grid for a stepped DMPS.

`V` are the **bin midpoints** (the setpoint voltages actually written to the
file). The function

1. converts the setpoints to mobilities,
2. constructs the matching bin edges,
3. calls [`DifferentialMobilityAnalyzers.setupDMAgridded`](@ref), which uses the
   **stepped** transfer function ``\\Omega`` evaluated at each setpoint.

The third point matters: a DMPS holds the voltage fixed and counts at every
step, so the kernel is the instantaneous transfer function at the setpoint. The
scan-averaged kernel ``\\Omega_{av}`` used by `setupSMPS`/`setupSMPSdata`
applies to a *scanning* SMPS, where the voltage ramps through the bin while the
counter integrates.
"""
function grid_from_voltages(Λ::DMAconfig, V::AbstractVector{<:Real})
    V = sort(float.(V))
    Z = vtoz(Λ, V)
    Ze = mobility_edges(Z)
    De = ztod.(Ref(Λ), 1, Ze)        # nm, ascending
    return setupDMAgridded(Λ, reverse(De))   # descending edges -> ΔlnD > 0
end

# ---------------------------------------------------------------
# Inversion and distributions
# ---------------------------------------------------------------

"""
    invert_scan(scan; ρ = 1.0, λ₁ = 0.1, λ₂ = 10.0, order = 0, initial = true, n = 1)

Invert one scan with Tikhonov regularization and return the number, volume and
mass distributions plus their moments.

Arguments
- `scan`: a [`Scan`](@ref)
- `ρ`: particle density [g cm⁻³] for the mass distribution
- `λ₁`, `λ₂`: search bounds for the regularization parameter
- `order`: Tikhonov order, 0, 1 or 2
- `initial`: use the ``\\mathbf{S}^{-1}\\mathbf{r}`` a-priori initial guess
- `n`: number of BLAS threads

With `ρ` in g cm⁻³ and `Dp` in nm the mass distribution is obtained in µg m⁻³.
"""
function invert_scan(
    scan::Scan;
    ρ::Real = 1.0,
    λ₁::Real = 0.1,
    λ₂::Real = 10.0,
    order::Integer = 0,
    initial::Bool = true,
    n::Integer = 1,
)
    Λ, V, N = scan.Λ, scan.V, scan.N

    # zero / non-finite responses are noise; the inversion requires r >= 0
    N = map(x -> (isfinite(x) && x > 0.0) ? x : 0.0, N)

    δ = grid_from_voltages(Λ, V)

    𝕣 = _response(scan.Dp, N, δ)

    𝕟 = rinv2(𝕣.N, δ; λ₁ = λ₁, λ₂ = λ₂, order = order, initial = initial, n = n)

    return _result(δ, 𝕟, ρ)
end

"""
    _response(Dp, N, δ)

Interpolate the measured response onto the DMA grid and return it as a
`SizeDistribution`.

`Dp`/`N` are the measured setpoints in any order, `δ` is the grid from
[`grid_from_voltages`](@ref). The evaluation points are clamped into the measured
range: by the construction of [`mobility_edges`](@ref) the outermost grid
midpoints *are* the outermost setpoints, and they can land a few ULPs outside the
range because `δ.Dp` and `Dp` are computed through different roundings. Without
the clamp, `extrapolate(..., 0.0)` silently zeroes the outermost bins, which
destroys the distribution tails.
"""
function _response(
    Dp::AbstractVector{<:Real},
    N::AbstractVector{<:Real},
    δ::DifferentialMobilityAnalyzer,
)
    o = sortperm(Dp)
    Dpₛ = Dp[o]
    itp = interpolate((Dpₛ,), N[o], Gridded(Linear()))
    etp = extrapolate(itp, 0.0)

    R = etp.(clamp.(δ.Dp, first(Dpₛ), last(Dpₛ)))
    return SizeDistribution([], δ.De, δ.Dp, δ.ΔlnD, R ./ δ.ΔlnD, R, :interpolated)
end

"""
    _result(δ, 𝕟, ρ)

Assemble the number/volume/mass distributions and moments from the inverted
size distribution.

The package works with ``dN/d\\ln D_p``; the ``1/\\log_{10}`` convention used
here introduces a factor ``\\ln 10``.
"""
function _result(δ::DifferentialMobilityAnalyzer, 𝕟::SizeDistribution, ρ::Real)
    # δ.Dp is descending (setupDMAgridded takes descending edges); present the
    # result sorted by ascending diameter
    o = reverse(axes(δ.Dp, 1))
    Dp = δ.Dp[o]
    N = 𝕟.N[o]                           # per-bin number concentration
    ln10 = log(10.0)

    dN = 𝕟.S[o] .* ln10                  # dN/dlog10Dp   [cm⁻³]

    # sphere volume, Dp [nm] -> [µm]
    vol = π / 6.0 .* (Dp ./ 1000.0) .^ 3      # µm³ per particle
    dV = vol .* dN                     # dV/dlog10Dp   [µm³ cm⁻³]
    dm = ρ .* dV                      # dm/dlog10Dp   [µg m⁻³]

    Ntot = sum(N)
    Vtot = sum(vol .* N)
    Mtot = ρ * Vtot

    DgN = _geomean(Dp, N)
    DgM = _geomean(Dp, ρ .* vol .* N)

    return DMPSResult(Dp, N, dN, dV, dm, Ntot, Vtot, Mtot, DgN, DgM)
end

""" geometric mean diameter of a weighted distribution; weights must be >= 0 """
function _geomean(Dp::AbstractVector{<:Real}, w::AbstractVector{<:Real})
    ok = isfinite.(w) .& (w .> 0.0) .& isfinite.(Dp) .& (Dp .> 0.0)
    any(ok) || return NaN
    ww = w[ok]
    return exp(sum(ww .* log.(Dp[ok])) / sum(ww))
end

"""
    distributions(res)

Return the distributions of a [`DMPSResult`](@ref) as a `DataFrame` with one row
per bin, ordered by ascending `Dp`.
"""
function distributions(res::DMPSResult)
    return DataFrame(
        Dp = res.Dp,
        N = res.N,
        dN_dlogDp = res.dN,
        dV_dlogDp = res.dV,
        dm_dlogDp = res.dm,
    )
end

"""
    moments(res)

Return the integral moments of a [`DMPSResult`](@ref) as a one-row `DataFrame`.
"""
function moments(res::DMPSResult)
    return DataFrame(
        Ntot = res.Ntot,
        Vtot = res.Vtot,
        Mtot = res.Mtot,
        DgN = res.DgN,
        DgM = res.DgM,
    )
end

# ---------------------------------------------------------------
# File-level entry points
# ---------------------------------------------------------------

"""
    parse_file(path; ρ = 1.0, kwargs...)

Invert every block of a DMPS file and write two CSVs per block into the file's
directory: `raw_<k>.csv` (setpoint voltage, concentration, diameter) and
`inverted_<k>.csv` (inverted distributions).

Any keyword accepted by [`read_dmps`](@ref) and [`invert_scan`](@ref) can be
passed through.
"""
function parse_file(path; ρ::Real = 1.0, kwargs...)
    outdir = dirname(path)
    scans = read_dmps(path; kwargs...)

    for (k, scan) in enumerate(scans)
        res = invert_scan(scan; ρ = ρ)
        raw = DataFrame(V = scan.V, N = scan.N, Dp = scan.Dp)
        CSV.write(joinpath(outdir, "raw_$(k).csv"), raw)
        CSV.write(joinpath(outdir, "inverted_$(k).csv"), distributions(res))
        println(
            "scan $k: Ntot = $(round(res.Ntot, digits = 2)) cm⁻³, DgN = $(round(res.DgN, digits = 2)) nm",
        )
    end

    return nothing
end

"""
    process_file(path; ρ = 1.0, plot = true, kwargs...)

Invert every block of a DMPS file and write `processed_data.csv` with the
integral moments of every scan. With `plot = true` also writes `banana.png`,
a heatmap of dN/dlog₁₀Dp over time (the banana plot).
"""
function process_file(path; ρ::Real = 1.0, plot::Bool = true, kwargs...)
    outdir = dirname(path)
    scans = read_dmps(path; kwargs...)

    isempty(scans) && error("no scans found in $path")

    rows = DataFrame(;
        t = DateTime[],
        Ntot = Float64[],
        Vtot = Float64[],
        Mtot = Float64[],
        DgN = Float64[],
        DgM = Float64[],
    )

    results = DMPSResult[]
    for scan in scans
        res = invert_scan(scan; ρ = ρ)
        push!(results, res)
        push!(
            rows,
            (scan.t0 + (scan.t1 - scan.t0) ÷ 2, res.Ntot, res.Vtot, res.Mtot, res.DgN, res.DgM),
        )
    end

    CSV.write(joinpath(outdir, "processed_data.csv"), rows)

    plot && banana_plot(scans, results; savepath = joinpath(outdir, "banana.png"))

    return rows
end

# ---------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------

"""
    banana_plot(scans, results; savepath = nothing, clims = nothing)

Heatmap of dN/dlog₁₀Dp as a function of time and diameter - the banana plot.

`scans` and `results` are the outputs of [`read_dmps`](@ref) and
[`invert_scan`](@ref). All scans are interpolated onto the diameter grid of the
first scan. Pass `clims = (lo, hi)` to fix the color scale in log₁₀ units.
"""
function banana_plot(
    scans::AbstractVector{Scan},
    results::AbstractVector{DMPSResult};
    savepath::Union{AbstractString,Nothing} = nothing,
    clims::Union{Tuple{Real,Real},Nothing} = nothing,
)
    isempty(scans) && error("no scans to plot")
    length(scans) == length(results) || error("scans and results must match")

    Dp0 = results[1].Dp
    nDp = length(Dp0)

    # minutes since the first scan, used as a continuous x-axis
    t0 = scans[1].t0
    tt = [
        Dates.value(scan.t0 + (scan.t1 - scan.t0) ÷ 2 - t0) / 60000.0 for
        scan in scans
    ]

    M = Matrix{Float64}(undef, nDp, length(scans))
    for (j, res) in enumerate(results)
        # all scans share the instrument, hence the same grid
        if res.Dp ≈ Dp0
            M[:, j] = res.dN
        else
            itp = interpolate((res.Dp,), res.dN, Gridded(Linear()))
            etp = extrapolate(itp, 0.0)
            M[:, j] = etp(Dp0)
        end
    end

    # log scale is required for aerosol size distributions spanning decades
    Mlog = log10.(max.(M, floatmin(Float64)))

    hi = clims === nothing ? maximum(Mlog) : clims[2]
    lo = clims === nothing ? hi - 4.0 : clims[1]
    Mlog = clamp.(Mlog, lo, hi)

    # GR renders heatmaps reliably without a display
    gr()
    p = heatmap(
        tt,
        Dp0,
        Mlog,
        yaxis = :log10,
        color = :viridis,
        clim = (lo, hi),
        xlabel = "time [min]",
        ylabel = "D_p [nm]",
        title = "dN/dlog₁₀D_p  [log₁₀ cm⁻³]",
        colorbar_title = "log₁₀ dN/dlogD_p",
    )

    savepath === nothing || savefig(p, savepath)
    return p
end

end
