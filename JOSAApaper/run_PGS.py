import pyramid_grid_solve2 as pgs
from multiprocessing import Pool
<<<<<<< HEAD
pool = Pool(2)
pool.map(pgs.grid_solve_pool, range(2))
=======
pool = Pool(20)
pool.map(pgs.grid_solve_pool, range(10))
>>>>>>> 7e4520d14d2fbdf0fb4aac30cda888f8b6aa7c34
