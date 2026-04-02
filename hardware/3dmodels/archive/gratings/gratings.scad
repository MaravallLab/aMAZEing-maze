//textures for the mice maze

plateX = 50;
plateY = 50;
plateZ = 2.4;

tol = 0.1;


grateX = plateX;
grateY = 3.5;
grateZ = 3.5;


module texture1(){
    for (i = [6:2*grateY: plateX-6]){
        translate([0, i, 0]){
            cube([grateX,grateY,plateZ+grateZ]);
        }//end translate
    }//end for 
    
    
    
    }
    
module texture2(){
    for (i = [grateY:2*grateY: plateY]){
        translate([5+tol, i, 0]){
            
            cube([grateX-10-tol,grateY,plateZ+grateZ]);
        }//end translate

    }//end for 
    
    
    
    }  
module texture3(){

difference(){

    intersection(){
    cube([plateX,plateY,plateZ+grateZ]);
        translate([0,-0,0]){
    rotate([0,0,45]){
        for (i = [-plateX:2*grateY: plateX]){
        translate([0, i, 0]){
            cube([grateX+50,grateY,plateZ+grateZ]);
        }//end translate
    }//end for 

    }//end rotate
}//end translate
}//end intersection

    translate([-tol,-tol,-tol]){
    cube([5+2*tol,plateY+2*tol,plateZ+grateZ+2*tol]);
    }//end translate
    translate([plateX-5-tol,-tol,-tol]){
    cube([5+2*tol,plateY+2*tol,plateZ+grateZ+2*tol]);
    }//end translate
}
}

translate([60,0,0]){
cube([plateX,plateY,plateZ]);    
texture1();
}


cube([plateX,plateY,plateZ]);    
    texture2();

translate([-60,0,0]){
cube([plateX,plateY,plateZ]);    
    texture3();
}