# RTKLIB解算过程结构体

## 1 sol_t

```c
typedef struct {        /* solution type */
    gtime_t time;       /* time (GPST) */
    double rr[6];       /* position/velocity (m|m/s) */
                        /* {x,y,z,vx,vy,vz} or {e,n,u,ve,vn,vu} */		//存储结算过程中的定位解，迭代更新
    float  qr[6];       /* position variance/covariance (m^2) */
                        /* {c_xx,c_yy,c_zz,c_xy,c_yz,c_zx} or */
                        /* {c_ee,c_nn,c_uu,c_en,c_nu,c_ue} */
    double dtr[6];      /* receiver clock bias to time systems (s) */
    unsigned char type; /* type (0:xyz-ecef,1:enu-baseline) */
    unsigned char stat; /* solution status (SOLQ_???) */
    unsigned char ns;   /* number of valid satellites */
    float age;          /* age of differential (s) */
    float ratio;        /* AR ratio factor for valiation */
} sol_t;
```

sol_t->rr在哪里被赋值

## 2 nav_t



## 3 rtk_t

rtk_t->prcopt 存储预定义处理配置
