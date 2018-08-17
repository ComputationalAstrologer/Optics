;+
;FUNCTION N=REFRACTIVITY_INFRARED(T,P,H,LAMBDA,/INTER)
;
;  Computes refractivity N=n-1 using Mathar's 2007 fit technique, for 
;  wet air with 372 ppmv of CO2, at temperature T (Celsius), atmsopsheric 
;  pressure P (Pa) and relative humidity H (%).
;
;  REFRACTIVITY IS NOT COMPUTED AND SET TO ZERO IN THE PHOTOMETRIC
;  BANDS.
;
;  Valid wavelength range of Mathar's model:
;  K-band  1.3  -  2.5 microns
;  L-band  2.8  -  4.2 microns
;  M-band  4.35 -  5.2 microns
;  N-band  7.5  - 14.1 microns
;  Q-band  16.0 - 24.0 microns
;
;  Under 1.3 microns and above 24 microns, we simply extrapolate
;  Mathar's model.
;  
;  Use the keyword /INTER to interpolate between the bands, if you
;  give an array of wavelengths for input LAMBDA and want a continuous
;  output array.
;
;INPUTS name | type | unit
;
;  T | scalar | celsius
;  air temperature in degree Celsius
;
;  P | scalar | Pascal
;  air pressure in pascals
;
;  H | scalar | %
;  relative humidity
;
;  LAMBDA | array/scalar real | microns
;  wavelengths.
;
;KEYWORD
;
;  /INTER set this keyword if you want to have a continuous N(lambda)
;  in-between the K,L,M,N,Q bands. This is not physical, of course,
;  but it's just there for convenience if you do not want discontinous
;  arrays.
;
;OUTPUT name | type | unit
;
;  N | array or scalar | 1
;  The refractivity, N=n-1
;
;HISTORY
;
;  Laurent Jolissaint, Leiden Observatory, June 14, 2008
;
;BUGS jolissaint@strw.leidenuniv.nl
;
;-
FUNCTION REFRACTIVITY_INFRARED,T,P,H,LAMBDA,INTER=INTER

  ; Mathar's fitting coefficients
  c_ref=[[ 0.00020019200,   0.00020004900,   0.00020002000,   0.00019988500,   0.00019943600], $
         [ 1.1347400d-10,   1.4522100d-10,   2.7534600d-10,   3.4473900d-10,   2.9912300d-09], $
         [-4.2459500d-15,   2.5095100d-13,   3.2570200d-13,  -2.7371400d-13,  -2.1486200d-11], $
         [ 1.0095700d-17,  -7.4583400d-16,  -6.9360300d-15,   3.9338300d-16,   1.4333800d-13], $
         [-2.9331500d-21,  -1.6143200d-18,   2.8561000d-18,  -5.6948800d-18,   1.2239800d-15], $
         [ 3.0722800d-25,   3.5278000d-21,   3.3875800d-19,   1.6455600d-20,  -1.1462800d-17]]
  c_T=  [[ 0.05886250000,   0.05884320000,   0.05900350000,   0.05939000000,   0.06217230000], $
         [-3.8576600d-08,  -8.2518200d-08,  -3.7576400d-07,  -1.7222600d-06,  -1.7707400d-05], $
         [ 8.8801900d-11,   1.3798200d-10,   1.3458500d-10,   2.3765400d-09,   1.5221300d-07], $
         [-5.6765000d-14,   3.5242000d-14,   1.2431600d-12,  -3.8181200d-12,  -9.5458400d-10], $
         [ 1.6661500d-17,  -7.3065100d-16,   5.0851000d-14,   3.0505000d-15,  -9.9670600d-12], $
         [-1.7484500d-21,  -1.6791100d-19,  -1.8924500d-16,  -1.5746400d-17,   9.2147600d-14]]
  c_TT= [[-3.01513000000,  -3.13579010000,  -4.09830000000,  -6.50355010000, -23.24090000000], $
         [ 0.00040616700,   0.00069412400,   0.00250037000,   0.01038300000,   0.10855700000], $
         [-5.1454400e-07,  -5.0060400e-07,   2.7518700e-07,  -1.3946400e-05,  -0.00102439000], $
         [ 3.4316100e-10,  -1.1666800e-09,  -6.5339800e-09,   2.2007700e-08,   6.3407200e-06], $
         [-1.0118900e-13,   2.0964400e-12,  -3.1058900e-10,  -2.7241200e-11,   7.6251700e-08], $
         [ 1.0674900e-17,   5.9103700e-15,   1.2774700e-12,   1.2636400e-13,  -6.7558700e-10]]
  c_H=  [[-1.0394500e-08,  -1.0814200e-08,  -1.4046300e-08,  -2.2193800e-08,  -7.7270700e-08], $
         [ 1.3685800e-12,   2.3010200e-12,   8.3935000e-12,   3.4737700e-11,   3.4723700e-10], $
         [-1.7103900e-15,  -1.5465200e-15,  -1.9092900e-15,  -4.6599100e-14,  -2.7267500e-12], $
         [ 1.1290800e-18,  -3.2301400e-18,  -1.2139900e-17,   7.3584800e-17,   1.7085800e-14], $
         [-3.2992500e-22,   6.3061600e-21,  -8.9886300e-19,  -8.9711900e-20,   1.5688900e-16], $
         [ 3.4474700e-26,   1.7388000e-23,   3.6466200e-21,   3.8081700e-22,  -1.5000400e-18]]
  c_HH= [[ 5.7325600e-13,   5.8681200e-13,   5.4360500e-13,   3.9352400e-13,  -3.2660400e-13], $
         [ 1.8636700e-17,   3.1219800e-17,   1.1280200e-16,   4.6408300e-16,   4.6360600e-15], $
         [-2.2815000e-20,  -1.9779200e-20,  -2.2997900e-20,  -6.2176400e-19,  -3.6427200e-17], $
         [ 1.5094700e-23,  -4.6194500e-23,  -1.9145000e-22,   9.8112600e-22,   2.2875600e-19], $
         [-4.4121400e-27,   7.8839800e-26,  -1.2035200e-23,  -1.2138400e-24,   2.0950200e-21], $
         [ 4.6120900e-31,   2.4558000e-28,   5.0095500e-26,   5.1511100e-27,  -2.0054700e-23]]
  c_P=  [[ 2.6708500e-09,   2.6690000e-09,   2.6689800e-09,   2.6680900e-09,   2.6682700e-09], $
         [ 1.3594100e-15,   1.6816200e-15,   2.7362900e-15,   6.9524700e-16,   1.2078800e-15], $
         [ 1.3529500e-19,   3.5307500e-18,   4.6346600e-18,   1.5907000e-18,   5.2264600e-18], $
         [ 8.1821800e-24,  -9.6345500e-21,  -9.1689400e-20,  -3.0345100e-21,   7.8302700e-20], $
         [-2.2295700e-27,  -2.2307900e-23,   1.3668500e-22,  -6.6148900e-23,   7.5323500e-22], $
         [ 2.4996400e-31,   4.5316600e-26,   4.1368700e-24,   1.7822600e-25,  -2.2881900e-25]]
  c_PP= [[ 6.0918600e-18,   6.0886000e-18,   6.1070600e-18,   6.1050800e-18,   6.1367500e-18], $
         [ 5.1902400e-24,   4.6156000e-23,   1.1662000e-22,   2.2769400e-23,   5.8549400e-23], $
         [-4.1947700e-28,   1.8428200e-25,   2.4473600e-25,   7.8632300e-26,   2.8605500e-25], $
         [ 4.3412000e-31,  -5.2447100e-28,  -4.9768200e-27,  -1.7444800e-28,   4.2519300e-27], $
         [-1.2244500e-34,  -1.2129900e-30,   7.4202400e-30,  -3.5979100e-30,   4.1345500e-29], $
         [ 1.3481600e-38,   2.4651200e-33,   2.2462500e-31,   9.7830700e-33,  -8.1294100e-33]]
  c_TH= [[ 4.9785900e-05,   5.1796200e-05,   6.7448800e-05,   0.00010677600,   0.00037597400], $
         [-6.6175200e-09,  -1.1214900e-08,  -4.0677500e-08,  -1.6851600e-07,  -1.7184900e-06], $
         [ 8.3203400e-12,   7.7650700e-12,   2.8906300e-12,   2.2620100e-10,   1.4670400e-08], $
         [-5.5179300e-15,   1.7256900e-14,   8.1989800e-14,  -3.5645700e-13,  -9.1723100e-11], $
         [ 1.6189900e-18,  -3.2058200e-17,   4.6838600e-15,   4.3798000e-16,  -9.5592200e-13], $
         [-1.6990100e-22,  -8.9943500e-20,  -1.9118200e-17,  -1.9454500e-18,   8.8050200e-15]]
  c_TP= [[ 7.7917600e-07,   7.7863800e-07,   7.7862700e-07,   7.7836800e-07,   7.7843600e-07], $
         [ 3.9649900e-13,   4.4639600e-13,   5.9329600e-13,   2.1640400e-13,   4.6184000e-13], $
         [ 3.9511400e-17,   7.8460000e-16,   1.4504200e-15,   5.8180500e-16,   3.0622900e-15], $
         [ 2.3358700e-21,  -1.9515100e-18,   4.8981500e-18,  -1.8961800e-18,  -6.2318300e-17], $
         [-6.3644100e-25,  -5.4208300e-21,   3.2794100e-20,  -1.9886900e-20,  -1.6111900e-19], $
         [ 7.1686800e-29,   1.0353000e-23,   1.2802000e-22,   5.8938100e-23,   8.0075600e-21]]
  c_HP= [[-2.0656700e-16,  -2.1724300e-16,  -2.1167600e-16,  -2.0636500e-16,  -2.7261400e-16], $
         [ 1.0614100e-21,   1.0474700e-21,   4.8792100e-21,   3.0023400e-20,   3.0466200e-19], $
         [-1.4998200e-24,  -5.2368900e-24,  -6.8254500e-24,  -4.2651900e-23,  -2.3959000e-21], $
         [ 9.8404600e-28,   8.1738600e-27,   9.4280200e-26,   6.8430600e-26,   1.4928500e-23], $
         [-2.8826600e-31,   3.0991300e-29,  -9.4642200e-28,  -4.6732000e-30,   1.3608600e-25], $
         [ 2.9910500e-35,  -3.6349100e-32,  -1.5368200e-30,   1.2611700e-31,  -1.3099900e-27]]

  ; reference temperature [K], pressure [Pa], humidity [%]
  Tref=290.65d
  Pref=75d3
  Href=10.d

  ; reference wave numbers [1/cm]
  sigmaK=1d4/2.35
  sigmaL=1d4/3.4
  sigmaM=1d4/4.8
  sigmaN=1d4/10.1
  sigmaQ=1d4/20.0

  ; functions of the fit
  DT  = 1.d/(T+273.15d)-1.d/Tref
  DTT = DT*DT
  DP  = P-Pref
  DPP = DP*DP
  DH  = H-Href
  DHH = DH*DH
  DTH = DT*DH
  DTP = DT*DP
  DHP = DH*DP

  ; indexes of K,L,M,N,Q windows
  wK=where(LAMBDA le  2.5)
  wL=where(LAMBDA ge  2.8  and LAMBDA le  4.2)
  wM=where(LAMBDA ge  4.35 and LAMBDA le  5.2)
  wN=where(LAMBDA ge  7.5  and LAMBDA le 14.1)
  wQ=where(LAMBDA ge 16.0)

  ; refractivity calculation
  sigma=1d4/LAMBDA ; [1/cm]
  refrac=dblarr(n_elements(LAMBDA))
  if wK[0] ne -1 then begin
    c_TPH=reform(c_ref[0,*]+c_T[0,*]*DT+c_TT[0,*]*DTT+c_H[0,*]*DH+c_HH[0,*]*DHH+c_P[0,*]*DP+c_PP[0,*]*DPP+c_TH[0,*]*DTH+c_TP[0,*]*DTP+c_HP[0,*]*DHP)
    refrac[wK]=c_TPH[0]
    for j=1,5 do refrac[wK]=refrac[wK]+c_TPH[j]*(sigma[wK]-sigmaK)^j
  endif
  if wL[0] ne -1 then begin
    c_TPH=reform(c_ref[1,*]+c_T[1,*]*DT+c_TT[1,*]*DTT+c_H[1,*]*DH+c_HH[1,*]*DHH+c_P[1,*]*DP+c_PP[1,*]*DPP+c_TH[1,*]*DTH+c_TP[1,*]*DTP+c_HP[1,*]*DHP)
    refrac[wL]=c_TPH[0]
    for j=1,5 do refrac[wL]=refrac[wL]+c_TPH[j]*(sigma[wL]-sigmaL)^j
  endif
  if wM[0] ne -1 then begin
    c_TPH=reform(c_ref[2,*]+c_T[2,*]*DT+c_TT[2,*]*DTT+c_H[2,*]*DH+c_HH[2,*]*DHH+c_P[2,*]*DP+c_PP[2,*]*DPP+c_TH[2,*]*DTH+c_TP[2,*]*DTP+c_HP[2,*]*DHP)
    refrac[wM]=c_TPH[0]
    for j=1,5 do refrac[wM]=refrac[wM]+c_TPH[j]*(sigma[wM]-sigmaM)^j
  endif
  if wN[0] ne -1 then begin
    c_TPH=reform(c_ref[3,*]+c_T[3,*]*DT+c_TT[3,*]*DTT+c_H[3,*]*DH+c_HH[3,*]*DHH+c_P[3,*]*DP+c_PP[3,*]*DPP+c_TH[3,*]*DTH+c_TP[3,*]*DTP+c_HP[3,*]*DHP)
    refrac[wN]=c_TPH[0]
    for j=1,5 do refrac[wN]=refrac[wN]+c_TPH[j]*(sigma[wN]-sigmaN)^j
  endif
  if wQ[0] ne -1 then begin
    c_TPH=reform(c_ref[4,*]+c_T[4,*]*DT+c_TT[4,*]*DTT+c_H[4,*]*DH+c_HH[4,*]*DHH+c_P[4,*]*DP+c_PP[4,*]*DPP+c_TH[4,*]*DTH+c_TP[4,*]*DTP+c_HP[4,*]*DHP)
    refrac[wQ]=c_TPH[0]
    for j=1,5 do refrac[wQ]=refrac[wQ]+c_TPH[j]*(sigma[wQ]-sigmaQ)^j
  endif

  ; interpolation between bands K,L,M,N,Q (there is no physics here, just an interpolation)
  if keyword_set(INTER) then begin
    wKL=where(LAMBDA gt  2.5 and LAMBDA lt  2.8)
    wLM=where(LAMBDA gt  4.2 and LAMBDA lt  4.35)
    wMN=where(LAMBDA gt  5.2 and LAMBDA lt  7.5)
    wNQ=where(LAMBDA gt 14.1 and LAMBDA lt 16.0)
    if wKL[0] ne -1 then begin
      s1=1d4/2.5 & s2=1d4/2.8
      c_TPH=reform(c_ref[0,*]+c_T[0,*]*DT+c_TT[0,*]*DTT+c_H[0,*]*DH+c_HH[0,*]*DHH+c_P[0,*]*DP+c_PP[0,*]*DPP+c_TH[0,*]*DTH+c_TP[0,*]*DTP+c_HP[0,*]*DHP)
      refrac1=c_TPH[0]
      for j=1,5 do refrac1=refrac1+c_TPH[j]*(s1-sigmaK)^j
      c_TPH=reform(c_ref[1,*]+c_T[1,*]*DT+c_TT[1,*]*DTT+c_H[1,*]*DH+c_HH[1,*]*DHH+c_P[1,*]*DP+c_PP[1,*]*DPP+c_TH[1,*]*DTH+c_TP[1,*]*DTP+c_HP[1,*]*DHP)
      refrac2=c_TPH[0]
      for j=1,5 do refrac2=refrac2+c_TPH[j]*(s2-sigmaL)^j
      refrac[wKL]=(refrac2-refrac1)/(s2-s1)*(sigma[wKL]-s1)+refrac1
    endif    
    if wLM[0] ne -1 then begin
      s1=1d4/4.2 & s2=1d4/4.35
      c_TPH=reform(c_ref[1,*]+c_T[1,*]*DT+c_TT[1,*]*DTT+c_H[1,*]*DH+c_HH[1,*]*DHH+c_P[1,*]*DP+c_PP[1,*]*DPP+c_TH[1,*]*DTH+c_TP[1,*]*DTP+c_HP[1,*]*DHP)
     refrac1=c_TPH[0]
      for j=1,5 do refrac1=refrac1+c_TPH[j]*(s1-sigmaL)^j
      c_TPH=reform(c_ref[2,*]+c_T[2,*]*DT+c_TT[2,*]*DTT+c_H[2,*]*DH+c_HH[2,*]*DHH+c_P[2,*]*DP+c_PP[2,*]*DPP+c_TH[2,*]*DTH+c_TP[2,*]*DTP+c_HP[2,*]*DHP)
      refrac2=c_TPH[0]
      for j=1,5 do refrac2=refrac2+c_TPH[j]*(s2-sigmaM)^j
      refrac[wLM]=(refrac2-refrac1)/(s2-s1)*(sigma[wLM]-s1)+refrac1
    endif    
    if wMN[0] ne -1 then begin
      s1=1d4/5.2 & s2=1d4/7.5
      c_TPH=reform(c_ref[2,*]+c_T[2,*]*DT+c_TT[2,*]*DTT+c_H[2,*]*DH+c_HH[2,*]*DHH+c_P[2,*]*DP+c_PP[2,*]*DPP+c_TH[2,*]*DTH+c_TP[2,*]*DTP+c_HP[2,*]*DHP)
      refrac1=c_TPH[0]
      for j=1,5 do refrac1=refrac1+c_TPH[j]*(s1-sigmaM)^j
      c_TPH=reform(c_ref[3,*]+c_T[3,*]*DT+c_TT[3,*]*DTT+c_H[3,*]*DH+c_HH[3,*]*DHH+c_P[3,*]*DP+c_PP[3,*]*DPP+c_TH[3,*]*DTH+c_TP[3,*]*DTP+c_HP[3,*]*DHP)
      refrac2=c_TPH[0]
      for j=1,5 do refrac2=refrac2+c_TPH[j]*(s2-sigmaN)^j
      refrac[wMN]=(refrac2-refrac1)/(s2-s1)*(sigma[wMN]-s1)+refrac1
    endif    
    if wNQ[0] ne -1 then begin
      s1=1d4/14.1 & s2=1d4/16.0
      c_TPH=reform(c_ref[3,*]+c_T[3,*]*DT+c_TT[3,*]*DTT+c_H[3,*]*DH+c_HH[3,*]*DHH+c_P[3,*]*DP+c_PP[3,*]*DPP+c_TH[3,*]*DTH+c_TP[3,*]*DTP+c_HP[3,*]*DHP)
      refrac1=c_TPH[0]
      for j=1,5 do refrac1=refrac1+c_TPH[j]*(s1-sigmaN)^j
      c_TPH=reform(c_ref[4,*]+c_T[4,*]*DT+c_TT[4,*]*DTT+c_H[4,*]*DH+c_HH[4,*]*DHH+c_P[4,*]*DP+c_PP[4,*]*DPP+c_TH[4,*]*DTH+c_TP[4,*]*DTP+c_HP[4,*]*DHP)
      refrac2=c_TPH[0]
      for j=1,5 do refrac2=refrac2+c_TPH[j]*(s2-sigmaQ)^j
      refrac[wNQ]=(refrac2-refrac1)/(s2-s1)*(sigma[wNQ]-s1)+refrac1
    endif    
  endif

  if n_elements(refrac) eq 1 then return,refrac[0]
  return,refrac

end
