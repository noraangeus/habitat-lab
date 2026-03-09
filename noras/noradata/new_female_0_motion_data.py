import numpy as np

start = np.array([0.0, 0.0, 0.0])
end   = np.array([1.0, 0.0, 2.0])

motion = {
    "fps": 30,
    "num_frames": 2,
    "displacement": [0.0, 1.0],

    "transform_array": [
        [
            [1,0,0,start[0]],
            [0,1,0,start[1]],
            [0,0,1,start[2]],
            [0,0,0,1]
        ],
        [
            [1,0,0,end[0]],
            [0,1,0,end[1]],
            [0,0,1,end[2]],
            [0,0,0,1]
        ]
    ],

    "transform_array2": [
        [
            [1,0,0,start[0]],
            [0,1,0,start[1]],
            [0,0,1,start[2]],
            [0,0,0,1]
        ],
        [
            [1,0,0,end[0]],
            [0,1,0,end[1]],
            [0,0,1,end[2]],
            [0,0,0,1]
        ]
    ],

    "joints_array": [
        [[0,0,0,1]],
        [[0,0,0,1]]
    ]
}