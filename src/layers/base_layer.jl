@inline function apply_act(::typeof(identity), x::Union{Number, AbstractArray})
    x
end

@inline function apply_act(activation::Function, x::Number)
    activation(x)
end

@inline function apply_act(activation::Function, x::AbstractArray)
    activation.(x)
end
