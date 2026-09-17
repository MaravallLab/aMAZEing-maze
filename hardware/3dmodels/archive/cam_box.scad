//cable hole diameter
cb_diameter = 20;

//base of camera board to top of lens
cam_h = 25;

//height camera should be suspended above the bottom of the box
cam_suspend_h = 30;

//side length of square indents the camera board will rest on
rest_indent_l = 7;

//whole camera board length. It is square
cam_board_l = 39;

//wall thickness of box x2
wall_thick = 6;

//length of box
box_l = cam_board_l+wall_thick;


//base square length
base_l = 70;

//added distance to get to middle of base square
base_add = (base_l-(cam_board_l+wall_thick))/2;

//subract cylinder from box with subracted box
translate([base_add, base_add, wall_thick])
{
difference()
{
// make cube which will be subtracted from bigger cube
difference()
{
{cube([cam_board_l+wall_thick, cam_board_l+wall_thick, cam_suspend_h+cam_h+wall_thick]);

translate([wall_thick/2, wall_thick/2, wall_thick/2])
{cube([cam_board_l, cam_board_l, cam_suspend_h+cam_h+5]);}

}}
// check how to rotate a cylinder

//make cylinder
//rotate by degrees around a defined axis

//define where to put hole
translate([box_l/2,box_l+1,20])
{rotate([90,0,0])
{cylinder(h=wall_thick/2+2, r1 = cb_diameter/2, r2 = cb_diameter/2);
}}}}


//change circularity of cylinder
$fn = 30;

//add indents for camera to rest on

// define thickness of the indent
indent_h = cam_suspend_h;

//module for supporting pillars
module pillar(){
    cube([rest_indent_l, rest_indent_l, indent_h]);
}

//module for m3 screw holes with a 0.1mm tolerance
module m3_hole(){
    diameter=3;
    tol = 0.1;
    cylinder(h=wall_thick*2, d1 = 
    diameter+tol*2, d2 = diameter+tol*2);
}

translate([base_add, base_add, wall_thick])
{
//make indent cube and move it to the corners of the box
translate([wall_thick/2,wall_thick/2,wall_thick/2])   
{difference(){
pillar();
 translate([rest_indent_l/2, rest_indent_l/2, cam_suspend_h-wall_thick*2+1]){    
 m3_hole();}}
}

//indent 2
translate([cam_board_l-wall_thick/1.5,wall_thick/2,wall_thick/2])
{difference(){
pillar();
 translate([rest_indent_l/2, rest_indent_l/2, cam_suspend_h-wall_thick*2+1]){    
 m3_hole();}}
}

//indent 3
translate([cam_board_l-wall_thick/1.5,cam_board_l-wall_thick/1.5,wall_thick/2])
{difference(){
pillar();
 translate([rest_indent_l/2, rest_indent_l/2, cam_suspend_h-wall_thick*2+1]){    
 m3_hole();}}
}

//indent 4
translate([wall_thick/2,cam_board_l-wall_thick/1.5,wall_thick/2])
{difference(){
pillar();
 translate([rest_indent_l/2, rest_indent_l/2, cam_suspend_h-wall_thick*2+1]){    
 m3_hole();}}
}
}

//how far in m3 holes should be
m3_loc = 5;

//make base plate and drill holes
difference()
{
cube([base_l, base_l, wall_thick]);
    
 translate([m3_loc, m3_loc, -1]){
 m3_hole();}

 translate([base_l-m3_loc, base_l-m3_loc,-1]){
 m3_hole();}

 translate([base_l-m3_loc, m3_loc,-1]){
 m3_hole();}

 translate([m3_loc, base_l-m3_loc, -1]){
 m3_hole();}
} 