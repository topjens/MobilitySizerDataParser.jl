using MobilitySizerDataParser
using DifferentialMobilityAnalyzers
using DataFrames
using Dates
using Test

# ------------------------------------------------------------------
# Instrument constants and synthetic-data helpers
# ------------------------------------------------------------------

const TSI = (r₁ = 9.37e-3, r₂ = 1.961e-2, l = 0.44369)

""" DMPS header line in the instrument's 15-value format """
function headerline(; qsh = 4.4, qsa = 1.0, p = 1000.0, t = 20.0)
    return join([qsh, qsa, TSI.r₁, TSI.r₂, TSI.l, 0, 0, p, 0, t, 0, 0, 0, 0, 0], " ")
end

""" DMAconfig assembled from the same header values parse_header reads """
function Λfromheader(; qsh = 4.4, qsa = 1.0, p = 1000.0, t = 20.0, leff = 0.0, polarity = :+, m = 6)
    return DMAconfig(
        t + 273.15,
        p * 100.0,
        qsa / 60.0 / 1000.0,
        qsh / 60.0 / 1000.0,
        TSI.r₁,
        TSI.r₂,
        TSI.l,
        leff,
        polarity,
        m,
        :cylindrical,
    )
end

""" write a DMPS file with the given (t0, t1, V, N) blocks """
function write_dmps(path, blocks)
    open(path, "w") do io
        for (t0, t1, V, N) in blocks
            println(
                io,
                "'$(Dates.format(t0, "mm-dd-yyyy HH:MM:SS"))' '$(Dates.format(t1, "mm-dd-yyyy HH:MM:SS"))'",
            )
            println(io, headerline())
            for (v, n) in zip(V, N)
                println(io, "$v $n")
            end
        end
    end
    return path
end

"""
Forward-model a distribution through the DMA and return `(V, N)` ready to be
written to a file: `V` ascending and `N` the response measured *at that voltage*.

`δ.𝐀 * 𝕟` returns the response in the grid's order (descending diameter), so it
must be reversed onto the ascending-voltage order a real instrument writes.
"""
function forward_block(Λ, δ, V, 𝕟, t0, dt = Minute(2))
    𝕣 = δ.𝐀 * 𝕟
    Vs = sort(V)
    o = sortperm(δ.Dp)              # ascending diameter, matching ascending V
    return (t0, t0 + dt, Vs, 𝕣.N[o])
end

@testset "MobilitySizerDataParser.jl" begin

    # --------------------------------------------------------------
    # Parsing
    # --------------------------------------------------------------
    @testset "header parsing" begin
        Λ = Λfromheader(qsh = 4.4, qsa = 1.0, p = 1000.0, t = 20.0)
        @test Λ.t ≈ 293.15
        @test Λ.p ≈ 1e5
        @test Λ.qsh ≈ 4.4 / 60.0 / 1000.0
        @test Λ.qsa ≈ 1.0 / 60.0 / 1000.0
        @test Λ.polarity == :+
        @test Λ.m == 6
        @test Λ.DMAtype == :cylindrical
    end

    @testset "read_dmps" begin
        path = tempname() * ".dat"
        V = exp10.(range(log10(50.0), stop = log10(9000.0), length = 30))
        write_dmps(path, [(DateTime(2024, 6, 1, 12, 0, 0), DateTime(2024, 6, 1, 12, 2, 0), V, ones(30))])

        scans = read_dmps(path)
        @test length(scans) == 1
        s = first(scans)
        @test s.t0 == DateTime(2024, 6, 1, 12, 0, 0)
        @test s.t1 == DateTime(2024, 6, 1, 12, 2, 0)
        @test length(s.V) == 30
        @test issorted(s.V)
        @test s.V[1] ≈ 50.0
        # ascending voltage means ascending diameter
        @test all(diff(s.Dp) .> 0)

        rm(path)
    end

    # --------------------------------------------------------------
    # Grid
    # --------------------------------------------------------------
    @testset "mobility_edges" begin
        # geometrically spaced midpoints: the interior edges are exact
        Z = exp10.(range(log10(1e-9), stop = log10(1e-5), length = 20))
        Ze = MobilitySizerDataParser.mobility_edges(Z)
        @test length(Ze) == 21
        @test all(Ze .> 0)
        @test sqrt(Ze[1] * Ze[2]) ≈ Z[1]
        @test sqrt(Ze[end-1] * Ze[end]) ≈ Z[end]
        for i in 2:length(Z)-1
            @test sqrt(Ze[i] * Ze[i+1]) ≈ Z[i]
        end
    end

    @testset "grid_from_voltages" begin
        Λ = Λfromheader()
        V = exp10.(range(log10(50.0), stop = log10(9000.0), length = 30))
        δ = grid_from_voltages(Λ, V)

        @test length(δ.Dp) == 30
        # ΔlnD > 0 requires the diameter edges to be descending
        @test all(δ.ΔlnD .> 0)

        # the grid midpoints must be the measured setpoints
        @test sort(δ.Z) ≈ sort(vtoz(Λ, V))

        # a monodisperse aerosol peaks at the setpoint it was measured at
        i = 15
        R = δ.𝐀 * DMALognormalDistribution([[1e6, δ.Dp[i], 1.01]], δ)
        @test argmax(R.N) == i
    end

    # --------------------------------------------------------------
    # Inversion against a known ground truth
    # --------------------------------------------------------------
    @testset "round trip" begin
        Λ = Λfromheader()
        V = exp10.(range(log10(20.0), stop = log10(9500.0), length = 60))
        δ = grid_from_voltages(Λ, V)

        𝕟 = DMALognormalDistribution([[400.0, 40.0, 1.4], [600.0, 150.0, 1.6]], δ)
        t0 = DateTime(2024, 6, 1, 12, 0, 0)

        path = tempname() * ".dat"
        write_dmps(path, [forward_block(Λ, δ, V, 𝕟, t0)])

        res = invert_scan(first(read_dmps(path)))

        @test sum(𝕟.N) ≈ res.Ntot rtol = 0.05

        ok = 𝕟.N .> 1e-6
        DgN_true = exp(sum(𝕟.N[ok] .* log.(𝕟.Dp[ok])) / sum(𝕟.N[ok]))
        @test DgN_true ≈ res.DgN rtol = 0.05

        @test length(res.dN) == length(res.Dp)
        @test all(res.dN .>= 0.0)
        @test issorted(res.Dp)

        rm(path)
    end

    # --------------------------------------------------------------
    # Volume, mass and moments
    # --------------------------------------------------------------
    @testset "volume and mass" begin
        Λ = Λfromheader()
        V = exp10.(range(log10(20.0), stop = log10(9500.0), length = 40))
        δ = grid_from_voltages(Λ, V)
        𝕟 = DMALognormalDistribution([[1000.0, 100.0, 1.5]], δ)
        t0 = DateTime(2024, 6, 1, 12, 0, 0)

        path = tempname() * ".dat"
        write_dmps(path, [forward_block(Λ, δ, V, 𝕟, t0)])
        res = invert_scan(first(read_dmps(path)); ρ = 1.5)

        @test all(isfinite.(res.dV))
        @test res.Mtot ≈ 1.5 * res.Vtot rtol = 1e-12

        # Vtot is the sum of N spheres of diameter Dp
        vol = π / 6.0 .* (res.Dp ./ 1000.0) .^ 3
        @test res.Vtot ≈ sum(vol .* res.N) rtol = 1e-9

        # mass is shifted to larger sizes than number
        @test res.DgM > res.DgN

        rm(path)
    end

    @testset "distributions and moments dataframes" begin
        Λ = Λfromheader()
        V = exp10.(range(log10(20.0), stop = log10(9500.0), length = 30))
        δ = grid_from_voltages(Λ, V)
        𝕟 = DMALognormalDistribution([[500.0, 80.0, 1.45]], δ)
        t0 = DateTime(2024, 6, 1, 12, 0, 0)

        path = tempname() * ".dat"
        write_dmps(path, [forward_block(Λ, δ, V, 𝕟, t0)])
        res = invert_scan(first(read_dmps(path)))

        df = distributions(res)
        @test names(df) == ["Dp", "N", "dN_dlogDp", "dV_dlogDp", "dm_dlogDp"]
        @test nrow(df) == length(res.Dp)

        m = moments(res)
        @test names(m) == ["Ntot", "Vtot", "Mtot", "DgN", "DgM"]
        @test nrow(m) == 1
        @test m.Ntot[1] ≈ res.Ntot

        rm(path)
    end

    # --------------------------------------------------------------
    # Multi-scan file + banana plot
    # --------------------------------------------------------------
    @testset "process_file and banana plot" begin
        Λ = Λfromheader()
        V = exp10.(range(log10(50.0), stop = log10(9000.0), length = 30))
        δ = grid_from_voltages(Λ, V)

        # a growing mode kept well inside the grid: the banana
        blocks = []
        for k in 0:5
            t0 = DateTime(2024, 6, 1, 12, 0, 0) + Minute(5k)
            𝕟 = DMALognormalDistribution([[500.0, 60.0 * 1.18^k, 1.4]], δ)
            push!(blocks, forward_block(Λ, δ, V, 𝕟, t0))
        end

        dir = mktempdir()
        path = joinpath(dir, "dmps.dat")
        write_dmps(path, blocks)

        rows = process_file(path; plot = true)
        @test nrow(rows) == 6
        @test issorted(rows.DgN)
        @test rows.DgN[1] < rows.DgN[end]

        @test isfile(joinpath(dir, "banana.png"))
        @test filesize(joinpath(dir, "banana.png")) > 10_000
        @test isfile(joinpath(dir, "processed_data.csv"))

        rm(dir; recursive = true)
    end
end
