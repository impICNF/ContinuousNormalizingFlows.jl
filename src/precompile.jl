@setup_workload begin
    @compile_workload begin
        fllprcmpltn = @load_preference("fullprecompilation", false)
        mts =
            fllprcmpltn ?
            Type{<:AbstractFlows}[
                RNODE,
                FFJORD,
                Planar,
                CondRNODE,
                CondFFJORD,
                CondPlanar,
            ] : Type{<:AbstractFlows}[RNODE]
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
            data_dist = Beta{data_type}(convert(Tuple{data_type, data_type}, (2, 4))...)
            r = convert.(data_type, rand(data_dist, nvars))
            if compute_mode <: MatrixMode
                r = hcat(r)
            end
            data_dist2 = Beta{data_type}(convert(Tuple{data_type, data_type}, (4, 2))...)
            r2 = convert.(data_type, rand(data_dist2, nvars))
            if compute_mode <: MatrixMode
                r2 = hcat(r2)
            end
            if mt <: AbstractCondICNF
                if mt <: CondPlanar
                    nn = PlanarLayer(nvars, tanh; n_cond = nvars)
                else
                    nn = Lux.Dense(2 * nvars => nvars, tanh)
                end
            else
                if mt <: Planar
                    nn = PlanarLayer(nvars, tanh)
                else
                    nn = Lux.Dense(nvars => nvars, tanh)
                end
            end
            icnf = construct(mt, nn, nvars; compute_mode)
            ps, st = Lux.setup(icnf.rng, icnf)
            ps = ComponentArray(ps)
            if mt <: AbstractCondICNF
                L = loss(icnf, omode, r, r2, ps, st)
            else
                L = loss(icnf, omode, r, ps, st)
            end
        end
    end
end
