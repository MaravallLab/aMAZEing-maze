platex = 50;
platey = 50;
platez = 3;

spoutHD = 2;

//tolerance
tol = 0.1;


module container(){
    difference(){
    cube([6,10,20]);
    
    translate([1,-2,1]){
        cube([4,10,20]);
        }//end translate
    }//end difference
    
}//end module
    
//difference(){
union(){
cube([platex,platey,platez]);

    translate([platex/2-3,2,0]){
    rotate([-90,0,0]){
    container();
    }//end rotate
}//end translate
}//end union


translate([platex/2,platey/2,-1]){
    cylinder(d=spoutHD,h=3*platez);
    
    }//end translate

translate([platex/2,5,-1]){
    rotate([30,0,0])
    cylinder(d=spoutHD,h=3*platez);
    
    }//end translate


//}