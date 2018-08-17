;+
;FUNCTION REFRACTIVITY(LAMBDA,[TC=,PA=,PW=,XCO2=,\WATER])
;
;    Computes with high precision the refractivity 
;    (i.e. N=n-1) of wet air or pure water. See REFERENCES.
;
;    ***********************************************************
;    PLEASE PAY ATTENTION TO THE INPUTS RANGES OF VALIDITY BELOW
;    ***********************************************************
;
;INPUTS name | type | units
;
;    LAMBDA | SCALAR OR ARRAY | MICROMETERS
;    Optical wavelength in the vaccum.
;    STRICT VALIDITY RANGE:
;        AIR:   0.3 to 1.7 MU-M
;        WATER: 0.2 to 1.9 MU-M
;
;OPTIONAL INPUTS name | type | units | default value if any
;
;  IF /WATER IS NOT SET:
;
;    TC | SCALAR | deg C | 0
;    Air temperature in degree Celsius.
;    VALIDITY RANGE: -40C to 100C. Default value set to typical 
;    night time temperature on a high altitude astronomical site.
;
;    PA | SCALAR | Pa | 70000 Pa (~ 0.7 atm)
;    Dry air pressure in Pascals. Default value correspond to a 
;    high altitude astronomical site (~3000 m).
;    VALIDITY RANGE: 10'000 to 140'000 Pa
;
;    PW | SCALAR | Pa | 0
;    Water vapor partial pressure in Pascals.
;    VALIDITY RANGE: 0 to 30'000 Pa
;
;    XCO2 | SCALAR | Part-per-million (ppm) | 450 (yr 2008)
;    Carbon dioxide content in ppm.
;    VALIDITY RANGE: 0 to 2'000 ppm
;
;  IF /WATER IS SET:
;
;    TC | SCALAR | deg C | 20
;    Water temperature in degree Celsius.
;    VALIDITY RANGE: -12C to 500C.
;
;    PA | SCALAR | 1000 Kg/m^3 | 1
;    Water density in tons/m^3.
;    VALIDITY RANGE 0 to 1.06.
;
;KEYWORDS
;
;    /WATER computes refractive index N=n-1 for pure water instead of 
;    air, at the wavelength LAMBDA.
;
;OUTPUT name | type | unit
;
;    N | DOUBLE SCALAR OR ARRAY | 1
;    Refractivity of air or water, N=n-1.
;
;REFERENCES
;
;    Index of Refraction of Air
;    Jack A. Stone and Jay H. Zimmerman
;    http://emtoolbox.nist.gov/Wavelength/Documentation.asp
;
;    Phillip E. Ciddor,
;    "Refractive index of air: new equations for the visible
;    and near infrared,"
;    Appl. Optics 35, 1566-1573 (1996).
;
;    The International Association for the Properties of Water and Steam
;    Erlangen, Germany September 1997
;    "Release on the Refractive Index of Ordinary Water Substance as a
;    Function of Wavelength, Temperature and Pressure"
;
;MODIFICATION HISTORY
;
;    Laurent Jolissaint, HIA/NRC, October 26, 2005, written.
;    May 02, 2008 LJ changed default ppm CO2 from 350 to 450.
;    Mar 07, 2014 LJ ()->[] for arrays
;
;BUGS jolissaint@strw.leidenuniv.nl
;
;-
FUNCTION REFRACTIVITY,LAMBDA,TC=TC,PA=PA,PW=PW,XCO2=XCO2,WATER=WATER

  ;arguments check-in
  if n_params() ne 1 then ERR_EXIT,FUN='REFRACTION_INDEX.PRO',$
    ERR='NEED THE OPTICAL WAVELENGTH'
  if not keyword_set(WATER) then begin
    if min(LAMBDA) lt 0.3 then ERR_EXIT,FUN='REFRACTION_INDEX.PRO',$
      ERR='OPTICAL WAVELENGTH OUT OF ALLOWED RANGE > 0.3 MU-M'
    if n_elements(TC) eq 0 then TC=0.d
    if n_elements(PA) eq 0 then PA=7d4
    if n_elements(PW) eq 0 then PW=0.d
    if n_elements(XCO2) eq 0 then XCO2=450.d
    if double(TC) lt -40 or double(TC) gt 100 then ERR_EXIT,FUN='REFRACTION_INDEX.PRO',$
      ERR='AIR TEMPERATURE OUT OF ALLOWED RANGE -40C ... +100C'
    if double(PA) lt 10000L or double(PA) gt 140000L then ERR_EXIT,FUN='REFRACTION_INDEX.PRO',$
      ERR='AIR PRESSURE OUT OF ALLOWED RANGE 10000 Pa ... 140000 Pa'
    if double(PW) lt 0 or double(PW) gt 30000L then ERR_EXIT,FUN='REFRACTION_INDEX.PRO',$
      ERR='WATER VAPOR PRESSURE OUT OF ALLOWED RANGE 0 Pa ... 30000 Pa'
    if double(XCO2) lt 0 or double(XCO2) gt 2000 then ERR_EXIT,FUN='REFRACTION_INDEX.PRO',$
      ERR='CO2 CONTENT OUT OF ALLOWED RANGE 0 ppm ... 2000 ppm'
  endif else begin
    if min(LAMBDA) lt 0.2 or max(LAMBDA) gt 1.9 then ERR_EXIT,FUN='REFRACTION_INDEX.PRO',$
      ERR='OPTICAL WAVELENGTH OUT OF ALLOWED RANGE 0.2 - 1.9 MU-M'
    if n_elements(TC) eq 0 then TC=20.d
    if double(TC) lt -12 or double(TC) gt 500 then ERR_EXIT,FUN='REFRACTION_INDEX.PRO',$
      ERR='WATER TEMPERATURE OUT OF ALLOWED RANGE -12C ... +500C'
    if n_elements(PA) eq 0 then PA=1.d
    if double(PA) lt 0 or double(PA) gt 1.06 then ERR_EXIT,FUN='REFRACTION_INDEX.PRO',$
      ERR='WATER DENSITY OUT OF ALLOWED RANGE 0 ... 1.06 ton/m^3'
  endelse

  if keyword_set(WATER) then goto,aqua

  ;empirical constants for Ciddor formula
  w=[2.952350d+02, 2.642200d+00,-3.2380d-02, 4.02800d-03]
  k=[2.380185d+02, 5.792105d+06, 5.7362d+01, 1.67917d+05]
  a=[1.581230d-06,-2.933100d-08, 1.1043d-10]
  b=[5.707000d-06,-2.051000d-08]
  c=[1.989800d-04,-2.376000d-06]
  d= 1.83d-11
  e=-7.65d-09
  P0     = 1.013250000d+05
  T0     = 2.881500000d+02
  Za     = 9.995922115d-01
  rho_vs = 9.859380000d-03
  R      = 8.314472000d+00
  M_v    = 1.801500000d-02
  aa=1.00062d
  bb=3.14d-08
  cc=5.6d-07

  ;water vapor mole fraction
  f=aa+bb*double(PA+PW)+cc*double(TC)^2
  xv=f*double(PW)/double(PA+PW)

  ;basic dispersion, from Ciddor's paper
  S=1.d/double(LAMBDA)^2
  r_as=1.000d-8*((k[1]/(k[0]-S))+(k[3]/(k[2]-S))) ; dispersion from dry air
  r_vs=1.022d-8*(w[0]+w[1]*S+w[2]*S^2+w[3]*S^3)   ; dispersion from water vapor

  ;correction factor for CO2
  M_a=2.89635d-2+1.2011d-8*(double(XCO2)-400.d)
  r_axs=r_as*(1+5.34d-7*(double(XCO2)-450.d))

  ;effect of water vapor
  TK=double(TC)+273.15d
  Z_m=1-(double(PA+PW)/TK)*(a[0]+a[1]*double(TC)+a[2]*double(TC)^2+(b[0]+b[1]*double(TC))*xv+(c[0]+c[1]*double(TC))*xv^2)+(double(PA+PW)/TK)^2*(d+e*xv^2)

  ;constituents densities
  rho_axs=P0*M_a/(Za*R*T0)
  rho_v=xv*double(PA+PW)*M_v/(Z_m*R*TK)
  rho_a=(1-xv)*double(PA+PW)*M_a/(Z_m*R*TK)

  ;air refractive index - 1
  n=(rho_a/rho_axs)*r_axs+(rho_v/rho_vs)*r_vs

  return,n

aqua:

  a0 = 2.44257733d-01
  a1 = 9.74634476d-03
  a2 =-3.73234996d-03
  a3 = 2.68678472d-04
  a4 = 1.58920570d-03
  a5 = 2.45934259d-03
  a6 = 9.00704920d-01
  a7 =-1.66626219d-02
  UV = 2.292020d-01
  IR = 5.432937d+00
  T0 = 273.15d
  L0 = 0.589d

  quo=a0+a1*double(PA)+a2*(double(TC)+T0)/T0+a3*(double(LAMBDA)/L0)^2*(double(TC)+T0)/T0+a4/(double(LAMBDA)/L0)^2+$
      a5/((double(LAMBDA)/L0)^2-(UV/L0)^2)+a6/((double(LAMBDA)/L0)^2-(IR/L0)^2)+a7*PA^2
  n=sqrt((2*quo+1)/(1-quo))

  return,n-1

end
