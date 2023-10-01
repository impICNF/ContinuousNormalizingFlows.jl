@setup_workload begin
    @compile_workload begin
        fllprcmpltn = @load_preference("fullprecompilation", false)
        mts =
            fllprcmpltn ? Type{<:AbstractICNF}[RNODE, FFJORD, Planar] :
            Type{<:AbstractICNF}[RNODE]
        cmts =
            fllprcmpltn ? Type{<:AbstractCondICNF}[CondRNODE, CondFFJORD, CondPlanar] :
            Type{<:AbstractCondICNF}[]
        compute_modes =
            fllprcmpltn ?
            Type{<:ComputeMode}[
                ADVecJacVectorMode,
                ADJacVecVectorMode,
                ZygoteVectorMode,
                SDVecJacMatrixMode,
                SDJacVecMatrixMode,
                ZygoteMatrixMode,
            ] : Type{<:ComputeMode}[ZygoteMatrixMode]
        omodes = Mode[TrainMode(), TestMode()]
        nvars = 1
        r = rand(Float32, nvars)
        r_arr = hcat(r)
        r2 = rand(Float32, nvars)
        r2_arr = hcat(r)

        for compute_mode in compute_modes, omode in omodes, mt in mts
            if mt <: Planar
                nn = PlanarLayer(nvars; use_bias = false)
            else
                nn = Lux.Dense(nvars => nvars; use_bias = false)
            end
            icnf = construct(mt, nn, nvars; compute_mode)
            ps, st = Lux.setup(icnf.rng, icnf)
            ps = ComponentArray(ps)
            if compute_mode <: VectorMode
                L = loss(icnf, omode, r, ps, st)
            elseif compute_mode <: MatrixMode
                L = loss(icnf, omode, r_arr, ps, st)
            end
        end
        for compute_mode in compute_modes, omode in omodes, mt in cmts
            if mt <: CondPlanar
                nn = PlanarLayer(nvars; use_bias = false, n_cond = nvars)
            else
                nn = Lux.Dense(2 * nvars => nvars; use_bias = false)
            end
            icnf = construct(mt, nn, nvars; compute_mode)
            ps, st = Lux.setup(icnf.rng, icnf)
            ps = ComponentArray(ps)
            if compute_mode <: VectorMode
                L = loss(icnf, omode, r, r2, ps, st)
            elseif compute_mode <: MatrixMode
                L = loss(icnf, omode, r_arr, r2_arr, ps, st)
            end
        end
    end
end
