These spectra given to me by Chris White, working with Mansi.
I ran these commands on the spectra to produce what is in the 2002cx directory

fixspec,'sn2002cx.p25.dat',redspec='sn2002cx.p26.dat',bluera=[3000,5425],redra=[5426,10000],outfile='sn2002cx.p25_long.dat'

mv sn2002cx.p25_long.dat sn2002cx.p25.dat

fixspec,'sn2005hk_b.m03.dat',redspec='sn2005hk_a.m03.dat',redscale=3.4,bluera=[3000,5000],redra=[5001,10000],outfile='sn2005hk.m03.dat'

rm sn2005hk_a.m03.dat
rm sn2005hk_b.m03.dat
rm sn2005hk_c.m03.dat

fixspec,'sn2005hk_a.m04.dat',redspec='sn2005hk_b.m04.dat',joinat=5250,outfile='sn2005hk.m04.dat'

rm sn2005hk_a.m04.dat
rm sn2005hk_b.m04.dat
rm sn2005hk_c.m04.dat
rm sn2005hk_d.m04.dat

rm sn2005hk_b.m05.dat
rm sn2005hk_c.m05.dat
mv sn2005hk_a.m05.dat sn2005hk.m05.dat

rm sn2005hk_b.m06.dat
rm sn2005hk_c.m06.dat
mv sn2005hk_a.m06.dat sn2005hk.m06.dat

rm sn2005hk_a.m07.dat
mv sn2005hk_b.m07.dat sn2005hk.m07.dat

rm sn2005hk_b.m08.dat
rm sn2005hk_c.m08.dat
mv sn2005hk_a.m08.dat sn2005hk.m08.dat

rm sn2005hk_b.p15.dat
mv sn2005hk_a.p15.dat sn2005hk.p15.dat

rm sn2005hk_b.p24.dat
rm sn2005hk_c.p24.dat
mv sn2005hk_a.p24.dat sn2005hk.p24.dat


fixspec,'sn2005hk_a.m04.dat',redspec='sn2005hk_b.m04.dat',joinat=5250,outfile='sn2005hk.m04.dat'

