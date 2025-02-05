using WunDeeDB
using Test

@testset "WunDeeDB.jl" begin
    @test new_add(2, 3) == 5
    @test new_add(-1, 1) == 0
    @test new_add(0, 0) == 0

    tmp = new_add(2.5,1)
    @test isa(tmp,Number) == true
end
