//length of cross section in makerbeam
mb_inner_w = 5.7;
//width of entrance to cross of maker beam
mb_ent_w = 3;
//depth of cross section in maker beam
mb_cross_d =2.5;

//start with rectangle for inner width
cube([mb_inner_w, mb_cross_d, 20]);

//ok now need rectangle coming out out the maker beam, but need to translate it
translate([(mb_inner_w-mb_ent_w)/2,mb_cross_d,8])
{cube([mb_ent_w, 1.5, 8]);
}

//next need empty circle/cylinder that will hold the test tube as a snug fit
//need to translate it to beyond the second cylinder as well
difference(){
translate([mb_inner_w/2,mb_cross_d+1.2+16.5/2,10])
{cylinder(h=3, d1=16.7, d2=17, center = false);
}
translate([mb_inner_w/2,mb_cross_d+1.2+16.5/2,10])
{cylinder(h=6, d1=16.4, d2=16, cantre=false);}
}