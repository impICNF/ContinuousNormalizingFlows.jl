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
        data_type = Float32

        for compute_mode in compute_modes, omode in omodes, mt in mts
            data_dist = Distributions.Beta{data_type}(
                convert(Tuple{data_type, data_type}, (2, 4))...,
            )
            r = convert.(data_type, rand(data_dist, nvars))
            if compute_mode <: MatrixMode
                r = hcat(r)
            end
            if mt <: Planar
                nn = PlanarLayer(nvars, tanh)
            else
                nn = Lux.Dense(nvars => nvars, tanh)
            end
            icnf = construct(mt, nn, nvars; compute_mode)
            ps, st = Lux.setup(icnf.rng, icnf)
            ps = ComponentArray(ps)
            L = loss(icnf, omode, r, ps, st)
        end
        for compute_mode in compute_modes, omode in omodes, mt in cmts
            data_dist = Distributions.Beta{data_type}(
                convert(Tuple{data_type, data_type}, (2, 4))...,
            )
            r = convert.(data_type, rand(data_dist, nvars))
            if compute_mode <: MatrixMode
                r = hcat(r)
            end
            data_dist2 = Distributions.Beta{data_type}(
                convert(Tuple{data_type, data_type}, (4, 2))...,
            )
            r2 = convert.(data_type, rand(data_dist2, nvars))
            if compute_mode <: MatrixMode
                r2 = hcat(r2)
            end
            if mt <: CondPlanar
                nn = PlanarLayer(nvars, tanh; n_cond = nvars)
            else
                nn = Lux.Dense(2 * nvars => nvars, tanh)
            end
            icnf = construct(mt, nn, nvars; compute_mode)
            ps, st = Lux.setup(icnf.rng, icnf)
            ps = ComponentArray(ps)
            L = loss(icnf, omode, r, r2, ps, st)
        end
    end
end
